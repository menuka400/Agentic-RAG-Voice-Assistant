"""
app/agent.py — LangGraph agent setup.

Graph architecture:
  START -> agent -> (tool_calls?) -> tools -> agent -> ... -> verify -> END

Nodes:
  agent  : LLM decides whether to answer directly or call a tool.
  tools  : Executes whichever tool(s) the LLM requested.
  verify : Lightweight self-check — confirms the draft answer actually addresses
           the user's question and uses the correct tool result. Regenerates once
           if the check fails; passes through immediately if correct.
"""

import logging
import re
import json
import uuid
from typing import List, TypedDict, Annotated

from langchain_core.messages import BaseMessage, SystemMessage, AIMessage, ToolMessage, HumanMessage
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode
from langchain_groq import ChatGroq

import config
from app.tools import web_search, get_current_datetime

# ── Logging setup ──────────────────────────────────────────────────────────

logger = logging.getLogger(__name__)

# ── 1. LLM & Tool Setup ───────────────────────────────────────────────────

if not config.GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is not set. Add it to your .env file.")

# Model is read from config.py — never hardcoded here.
# Currently: llama-3.3-70b-versatile (free tier, far better tool-calling than 8b).
llm = ChatGroq(
    model=config.GROQ_MODEL,
    api_key=config.GROQ_API_KEY,
    temperature=0,
)

# Tool registry. To add a new tool (e.g., RAG), just append it here.
tools = [web_search, get_current_datetime]
llm_with_tools = llm.bind_tools(tools)


# ── 2. Consolidated System Prompt ─────────────────────────────────────────
#
# One single system message is injected per request. It merges the base
# persona (from routes.py) with all tool-selection and answer-quality rules.
# There are NO competing system messages in the same invocation.

AGENT_SYSTEM_PROMPT = """{base_prompt}

━━━ TOOL SELECTION RULES ━━━

CONVERSATIONAL MESSAGES
If the user's most recent message is a greeting, small talk, or simple phrase
(e.g. "hi", "hello", "thanks", "bye", "how are you"), respond directly WITHOUT
calling any tool — regardless of what was discussed earlier in the conversation.
Never carry over tool-calling intent from a previous turn.

CURRENT TIME / DATE QUESTIONS
For any question asking for the current time or date in a specific place:
- ALWAYS use the `get_current_datetime` tool. It is fast, local, and accurate.
- You MUST convert the user's location into the correct IANA timezone string
  BEFORE calling the tool (e.g. "Sri Lanka" → "Asia/Colombo", "India" → "Asia/Kolkata",
  "New York" → "America/New_York", "Japan" → "Asia/Tokyo").
- Do NOT use web_search for time/date questions.
- Do NOT answer the time from memory — always call the tool.

REAL-WORLD FACTS, EVENTS, AND LIVE DATA
For ANY question about facts, outcomes, results, prices, scores, or events that
could have changed after your training cutoff:
- ALWAYS use the `web_search` tool. Do NOT answer from memory.
- This includes: sports results ("who won X"), election outcomes, exchange rates,
  current prices, recent news, competition winners, company stock prices, etc.
- Preserve the EXACT direction and intent of the user's question. For example,
  if the user asks "1 USD to LKR", search for "USD to LKR" and report how many
  LKR you get for 1 USD — do NOT invert the direction and report LKR to USD.

━━━ ANSWER QUALITY RULES ━━━

USING TOOL OUTPUT
- If a tool was called in the current response cycle and returned a result,
  your final answer MUST be based on that exact returned value.
- Do NOT say "I don't have access to real-time data" or mention knowledge
  cutoffs when a tool result is already present in this turn.
- Only use tool results that were returned in DIRECT response to the CURRENT
  user message. Do not reuse or reference tool results from earlier turns
  if the topic has changed.

ACCURACY & PRECISION
- State the exact values, numbers, or facts returned by the tool.
- Do not summarize, round, or paraphrase tool output in a way that loses
  meaningful precision.
- If no tool result is available and the answer is genuinely uncertain, say so.
"""


# ── 3. Graph Nodes ─────────────────────────────────────────────────────────

def call_model(state: MessagesState):
    """
    Agent node: invokes the LLM. Returns an AIMessage that is either a plain
    text final answer, or a tool-call request that routes to the tools node.

    Error handling: if Groq rejects the response due to a malformed tool call
    (the `tool_use_failed` / `failed_generation` error pattern), we attempt to
    salvage the intended call via regex extraction before falling back to a retry.
    """
    logger.info("Agent node — state has %d messages.", len(state["messages"]))
    try:
        response = llm_with_tools.invoke(state["messages"])
        logger.debug(
            "LLM response: type=%s content=%r tool_calls=%s",
            response.type,
            response.content[:100] if response.content else "",
            getattr(response, "tool_calls", []),
        )
        return {"messages": [response]}

    except Exception as e:
        err_str = str(e).lower()
        if (
            "failed to call a function" in err_str
            or "tool_use_failed" in err_str
            or "failed_generation" in err_str
        ):
            logger.warning("Groq tool_use_failed error detected. Attempting regex fallback.")

            # Attempt to parse the raw text tool call from the error's failed_generation
            match = re.search(r"<function=(\w+)>(\{.*?\})", str(e))
            if match:
                func_name = match.group(1)
                try:
                    func_args = json.loads(match.group(2))
                    logger.info("Regex fallback succeeded: %s(%s)", func_name, func_args)
                    # Synthesize a valid AIMessage with tool_calls so the graph
                    # routes to the ToolNode exactly as if the LLM had called it properly.
                    return {
                        "messages": [
                            AIMessage(
                                content="",
                                tool_calls=[
                                    {
                                        "name": func_name,
                                        "args": func_args,
                                        "id": f"call_{uuid.uuid4().hex[:8]}",
                                    }
                                ],
                            )
                        ]
                    }
                except json.JSONDecodeError:
                    logger.warning("Regex fallback JSON parse failed.")

            logger.info("Retrying LLM call once.")
            try:
                response = llm_with_tools.invoke(state["messages"])
                return {"messages": [response]}
            except Exception as retry_e:
                logger.error("Retry also failed: %s. Returning graceful fallback.", retry_e)
                return {
                    "messages": [
                        AIMessage(
                            content="Sorry, I had trouble processing that request — could you rephrase your question?"
                        )
                    ]
                }

        # Non-tool-call errors (auth, rate limit, etc.) — let run_agent catch them.
        raise e


def should_continue(state: MessagesState) -> str:
    """
    Conditional edge: after the agent node, go to tools if tool calls were
    requested, or proceed to the verify node for a final answer.
    """
    last = state["messages"][-1]
    if getattr(last, "tool_calls", None):
        return "tools"
    return "verify"


def verify_answer(state: MessagesState):
    """
    Verification node: runs after the agent produces a draft final answer.

    Strategy: run a brief verification check ONLY if a ToolMessage is present
    in the current graph run (i.e. a tool was actually called this turn).
    If no tools were called, there is nothing to cross-check — pass through.

    The verification prompt asks the LLM to confirm:
      - Did you answer the actual question asked?
      - Does your answer preserve the correct direction/intent (e.g. USD→LKR)?
      - Does your answer match the tool's returned value exactly?
      - Are you stating anything from memory that should have come from the tool?

    If the check yields "CORRECT", return the draft as-is (no extra LLM call).
    If the check yields a correction, return the corrected answer instead.
    """
    messages = state["messages"]
    draft = messages[-1]

    # Find the most recent HumanMessage (current user query)
    user_question = ""
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            user_question = m.content
            break

    # Collect ToolMessages that were generated in THIS graph run.
    # They appear AFTER the last HumanMessage in the message list.
    tool_results = []
    past_human = False
    for m in messages:
        if isinstance(m, HumanMessage):
            past_human = True
            tool_results = []  # reset on each human turn; only keep the latest
        elif past_human and isinstance(m, ToolMessage):
            tool_results.append(m.content)

    # Skip verification if no tools were called this turn (nothing to cross-check).
    if not tool_results:
        logger.info("Verify node — no tool results this turn, passing through.")
        return {}  # no-op: state unchanged

    tool_results_str = "\n".join(f"- {r}" for r in tool_results)
    logger.info("Verify node — checking draft answer against %d tool result(s).", len(tool_results))

    verify_prompt = f"""You are a quality-check assistant. Evaluate the following draft answer.

USER QUESTION: {user_question}

TOOL RESULTS RETURNED THIS TURN:
{tool_results_str}

DRAFT ANSWER: {draft.content}

Check the draft answer for these problems:
1. Does it actually answer the user's question (not a related but different question)?
2. Does it preserve the correct direction/intent (e.g. if asked "USD to LKR", does it give LKR per USD, not USD per LKR)?
3. Does it use the exact values from the tool results (not made-up numbers)?
4. Does it avoid incorrectly saying "I don't have real-time data" when tool data IS available?

IMPORTANT: Trust the tool results completely. Do not second-guess them based on your training data.

If the draft answer is correct on all points, reply with exactly: CORRECT
If any point fails, reply with a single, clean, corrected answer. 
CRITICAL: Do NOT include any meta-commentary, explanations, or internal reasoning (e.g. do not say "The draft is incorrect because..."). Output ONLY the final user-facing text.
"""

    verification = llm.invoke([SystemMessage(content=verify_prompt)])
    corrected = verification.content.strip()

    if corrected.upper() == "CORRECT":
        logger.info("Verify node — draft answer approved.")
        return {}  # pass through unchanged
    else:
        logger.info("Verify node — draft corrected: %r", corrected[:80])
        # Return an AIMessage with the SAME id as the draft so LangGraph replaces it
        # rather than appending it alongside the original draft.
        return {"messages": [AIMessage(content=corrected, id=draft.id)]}


# ── 4. Build and Compile Graph ─────────────────────────────────────────────

graph_builder = StateGraph(MessagesState)

graph_builder.add_node("agent", call_model)
graph_builder.add_node("tools", ToolNode(tools))
graph_builder.add_node("verify", verify_answer)

graph_builder.add_edge(START, "agent")
graph_builder.add_conditional_edges("agent", should_continue, ["tools", "verify"])
graph_builder.add_edge("tools", "agent")   # loop back after tool executes
graph_builder.add_edge("verify", END)      # verification is the final step

agent_app = graph_builder.compile()


# ── 5. Public API ──────────────────────────────────────────────────────────

async def run_agent(messages: List[BaseMessage], system_prompt: str = None) -> str:
    """
    Execute the agent graph asynchronously.

    Args:
        messages: Full conversation history as LangChain BaseMessage objects
                  (Human/AI pairs only — no raw ToolMessages from prior turns).
        system_prompt: Optional base persona/instruction to merge into the
                       consolidated system prompt.

    Returns:
        The final text response from the agent.
    """
    full_messages = messages.copy()

    # Single consolidated SystemMessage — merges base persona with all rules.
    base = system_prompt or "You are a helpful AI assistant with access to tools."
    full_messages.insert(0, SystemMessage(content=AGENT_SYSTEM_PROMPT.format(base_prompt=base)))

    logger.info("Starting graph invocation with %d messages.", len(full_messages))
    for i, m in enumerate(full_messages):
        logger.debug("  [%d] %s: %r", i, m.type.upper(), m.content[:80] if m.content else "")

    try:
        result = await agent_app.ainvoke({"messages": full_messages})

        final_message = result["messages"][-1]

        # Log tool results for monitoring (not debug noise, useful for prod review)
        for i, m in enumerate(result["messages"]):
            if isinstance(m, ToolMessage):
                logger.info("Tool result [idx %d] id=%s: %r", i, m.tool_call_id, m.content[:120])

        logger.info(
            "Graph complete — final message type=%s content=%r",
            final_message.type.upper(),
            final_message.content[:100],
        )

        return final_message.content

    except Exception as exc:
        err_str = str(exc).lower()
        if "authentication" in err_str or "api key" in err_str:
            raise RuntimeError("Invalid Groq API key. Check your .env file.")
        elif "rate limit" in err_str:
            raise RuntimeError("Groq rate limit reached. Please wait and try again.")
        else:
            raise RuntimeError(f"Agent error: {exc}")
