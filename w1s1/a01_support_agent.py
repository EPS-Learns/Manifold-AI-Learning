"""
SESSION 1 PROOF OF PROGRESS: Conversational Support Agent v1
============================================================
A production-aware support agent that combines everything from Day 1:
  - Configurable persona (SystemMessage)
  - chat() function with conversation memory
  - Error handling (specific exceptions, not a bare except)
  - Structured JSON logging with latency + token usage
  - A 4-turn simulated conversation + a token-cost summary

Anchor: A demo gives a response. A system gives a trace.

Run:  python assignments/a01_support_agent.py

Reflection to submit (5 lines):
  1. What did I build?
  2. Where is memory handled?
  3. Where can this fail?
  4. What did I add to make it production-aware?
  5. What will I improve next?
"""

import json
import logging
import time
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

load_dotenv()

# ---------- logging ----------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("support_agent")

# ---------- configurable persona ----------
COMPANY = "AI EduTech"  # change this to see how the agent's persona changes
PERSONA = (
    f"You are a customer support agent for {COMPANY}, an edtech company focusing on AI education. "
    "Be helpful, professional, and concise. Answer in 2-3 sentences max."
    "Be Human and empathetic in your tone, and use the customer's name if provided."
)

llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash", temperature=0.3)


# conversation memory (owned by THIS application)
conversation = [SystemMessage(content=PERSONA)]

# running totals for the cost summary
totals = {"input_tokens": 0, "output_tokens": 0, "turns": 0}


def chat(user_input: str, session_id: str = "sess-001") -> str:
    """One support turn: append input, call LLM with logging + error handling, store reply."""
    start = time.time()
    conversation.append(HumanMessage(content=user_input))

    try:
        response = llm.invoke(conversation)
    except ValueError as e:
        logger.error(json.dumps({"event": "llm_error", "type": "ValueError", "msg": str(e)}))
        return "Sorry, that input couldn't be processed."
    except TimeoutError as e:
        logger.error(json.dumps({"event": "llm_error", "type": "Timeout", "msg": str(e)}))
        return "The request timed out. Please try again."
    except Exception as e:  # last-resort fallback — the user never sees a stack trace
        logger.error(json.dumps({"event": "llm_error", "type": type(e).__name__, "msg": str(e)}))
        return "Something went wrong. Our team has been notified."

    if not response.content :
        logger.warning(json.dumps({"event": "empty_response", "session_id": session_id}))
        return "I couldn't generate a response. Please try again."

    conversation.append(response)  # store the agent's reply -> memory continuity

    latency_ms = round((time.time() - start) * 1000, 2)
    usage = response.usage_metadata or {}
    totals["input_tokens"] += usage.get("input_tokens", 0)
    totals["output_tokens"] += usage.get("output_tokens", 0)
    totals["turns"] += 1

    logger.info(json.dumps({
        "event": "turn_success",
        "session_id": session_id,
        "turn": totals["turns"],
        "latency_ms": latency_ms,
        "input_tokens": usage.get("input_tokens", 0),
        "output_tokens": usage.get("output_tokens", 0),
    }))
    return response.content

def content_to_text(content):
    if isinstance(content, str):
        return content
    return "".join(
        block["text"]
        for block in content
        if isinstance(block, dict) and block.get("type") == "text"
    )

def main() -> None:
    print("=" * 60)
    print(f"{COMPANY} Support Agent v1")
    print("=" * 60)

    for user_msg in [
        "Hi, what is the price of the course?",
        "How long is the trial window?",
        "I want to buy the course, when is the last date to register for it?",
        "Great. I will get back to you soon. Thanks for the help!",
    ]:
        reply = content_to_text(chat(user_msg))
        print("\n" + "-" * 60)
        print(f"User:  {user_msg}")
        print(f"Agent: {reply}")
        print("-" * 60 + "\n")

    print("\n" + "=" * 60)
    print("TOKEN / COST SUMMARY")
    print(json.dumps(totals, indent=2))
    #print("A system gives a trace — you can now answer: how many tokens did this session cost?")


if __name__ == "__main__":
    main()








'''
Reflection:

1 What did I build?
    I built a conversational support agent with a configurable persona, conversation memory, error handling, and structured logging. 

2 Where is memory handled?
    Memory is handled by the application itself, specifically in the 'conversation' list that stores the sequence of messages between the user and the agent. 
    Each user input and agent response is appended to this list, allowing the agent to maintain context across turns.

3 Where can this fail?
    Errors not handled by the try-except blocks
    Giving irrelevant answers since there is no detailed info about the company given to the model.

4 What did I add to make it production-aware?
    Adding Memory to maintain context across turns.
    Structured logging for monitoring and debugging.
    Error handling for graceful degradation.

5 What will I improve next?
    I will improve the agent's ability to handle more complex queries by providing it with more detailed information about the company and its offerings.
    I will explore ways to optimize token usage to reduce costs while maintaining response quality.

'''