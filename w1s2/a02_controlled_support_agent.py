"""
SESSION 2 PROOF OF PROGRESS: Controlled Support Agent v1
========================================================
Combines everything from Day 2:
  - A custom @tool with a clear docstring (a capability boundary)
  - A LangGraph workflow with State, Nodes, Edges
  - State tracks customer_id, customer_tier, issue_category, final_response
  - At least one conditional route (here: tier + technical + verification)
  - Two different inputs take two different paths
  - The path taken is LOGGED

Graceful degradation: if the LLM call fails (or no key), it falls back to
keyword classification so the graph still runs end-to-end (Day 1 lens).

Anchor: Tools give capability. Graphs give control.

Setup:  pip install langgraph
Run:    python assignments/a02_controlled_support_agent.py

Reflection to submit (5 lines):
  1. What capability did I give the agent?
  2. What tool did I create and when should it be used?
  3. What state fields did my graph track?
  4. What routing decision did my graph make?
  5. Why is this more controlled than a simple agent?
"""

import json
import logging
import re
from typing import TypedDict

from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger("controlled_support_agent")

# LLM is optional — load lazily so the file runs even without a key.
try:
    from langchain_openai import ChatOpenAI
    #_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    _llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash", temperature=0.3)

except Exception:  # missing package/key -> we'll use the keyword fallback
    _llm = None


# ===== STATE =====
class SupportState(TypedDict):
    user_message: str
    customer_id: str
    customer_tier: str        # 'premium' | 'standard' | 'unknown'
    issue_category: str       # 'billing' | 'technical' | 'general'
    route_taken: str
    ticket_id: str
    final_response: str


# ===== CUSTOM TOOL (capability boundary) =====
@tool
def create_priority_ticket(customer_id: str, issue_category: str, priority: str) -> str:
    """
    Create a support ticket for a customer.

    Use this whenever a support request needs to be tracked and assigned —
    for billing disputes, technical failures, or any issue that needs follow-up.

    Args:
        customer_id: The customer's ID (e.g. 'CUST-001')
        issue_category: 'billing', 'technical', or 'general'
        priority: 'low', 'normal', or 'high'
    """
    import random
    ticket_id = f"TKT-{random.randint(10000, 99999)}"
    logger.info(json.dumps({"event": "ticket_created", "ticket_id": ticket_id,
                            "customer_id": customer_id, "priority": priority}))
    return ticket_id


# ===== NODES =====
KNOWN_CUSTOMERS = {"CUST-001": "premium", "CUST-003": "premium", "CUST-555": "standard"}


def identify_customer(state: SupportState) -> dict:
    match = re.search(r"CUST-\d+", state["user_message"].upper())
    customer_id = match.group(0) if match else "CUST-UNKNOWN"
    tier = KNOWN_CUSTOMERS.get(customer_id, "unknown")
    return {"customer_id": customer_id, "customer_tier": tier}


def classify_issue(state: SupportState) -> dict:
    text = state["user_message"]
    valid = {"billing", "technical", "general"}
    category = None
    if _llm is not None:
        try:
            resp = _llm.invoke([
                SystemMessage(content="Classify the support issue into exactly one of: "
                                      "billing, technical, general. Return ONLY the word."),
                HumanMessage(content=text),
            ])
            cand = resp.content.strip().lower()
            category = cand if cand in valid else None
        except Exception as e:
            logger.warning(json.dumps({"event": "classify_fallback", "error": type(e).__name__}))
    if category is None:  # keyword fallback (graceful degradation)
        low = text.lower()
        if any(w in low for w in ["refund", "charge", "billing", "payment"]):
            category = "billing"
        elif any(w in low for w in ["error", "bug", "broken", "login", "crash"]):
            category = "technical"
        else:
            category = "general"
    return {"issue_category": category}


def verify_path(state: SupportState) -> dict:
    return {"route_taken": "verification",
            "final_response": "We couldn't find your account. Please share a valid "
                              "customer ID (e.g. CUST-123) so we can verify you."}


def technical_path(state: SupportState) -> dict:
    ticket = create_priority_ticket.invoke({"customer_id": state["customer_id"],
                                            "issue_category": "technical", "priority": "high"})
    return {"route_taken": "technical", "ticket_id": ticket,
            "final_response": f"Routed to technical support. Ticket {ticket} created; an "
                              "engineer will follow up with troubleshooting steps."}


def premium_path(state: SupportState) -> dict:
    ticket = create_priority_ticket.invoke({"customer_id": state["customer_id"],
                                            "issue_category": state["issue_category"],
                                            "priority": "high"})
    return {"route_taken": "premium", "ticket_id": ticket,
            "final_response": f"Premium {state['issue_category']} issue prioritized. Ticket "
                              f"{ticket}; a senior agent will contact you within 1 hour."}


def standard_path(state: SupportState) -> dict:
    ticket = create_priority_ticket.invoke({"customer_id": state["customer_id"],
                                            "issue_category": state["issue_category"],
                                            "priority": "normal"})
    return {"route_taken": "standard", "ticket_id": ticket,
            "final_response": f"Your {state['issue_category']} issue is logged as {ticket}. "
                              "We'll respond within 24 hours."}


# ===== CONDITIONAL ROUTE =====
def route_request(state: SupportState) -> str:
    if state["customer_tier"] == "unknown":
        return "verify_path"
    if state["issue_category"] == "technical":
        return "technical_path"
    return "premium_path" if state["customer_tier"] == "premium" else "standard_path"


# ===== BUILD =====
def build_graph():
    b = StateGraph(SupportState)
    for name, fn in [
        ("identify_customer", identify_customer),
        ("classify_issue", classify_issue),
        ("verify_path", verify_path),
        ("technical_path", technical_path),
        ("premium_path", premium_path),
        ("standard_path", standard_path),
    ]:
        b.add_node(name, fn)
    b.set_entry_point("identify_customer")
    b.add_edge("identify_customer", "classify_issue")
    b.add_conditional_edges("classify_issue", route_request, {
        "verify_path": "verify_path",
        "technical_path": "technical_path",
        "premium_path": "premium_path",
        "standard_path": "standard_path",
    })
    for end_node in ["verify_path", "technical_path", "premium_path", "standard_path"]:
        b.add_edge(end_node, END)
    return b.compile()


def run(graph, message: str, label: str) -> None:
    print("\n" + "=" * 60)
    print(label)
    print("=" * 60)
    result = graph.invoke({
        "user_message": message, "customer_id": "", "customer_tier": "",
        "issue_category": "", "route_taken": "", "ticket_id": "", "final_response": "",
    })
    logger.info(json.dumps({
        "event": "request_routed",
        "customer_id": result["customer_id"],
        "tier": result["customer_tier"],
        "category": result["issue_category"],
        "route_taken": result["route_taken"],
    }))
    print(f"ROUTE : {result['route_taken'].upper()}")
    print(f"REPLY : {result['final_response']}")


if __name__ == "__main__":
    graph = build_graph()
    # Two different inputs taking two different paths:
    run(graph, "CUST-001 here, I was double-charged for my plan.", "TEST 1: premium + billing")
    run(graph, "ID CUST-555 — the app keeps crashing on login.", "TEST 2: technical issue")
    run(graph, "Hi, I need help but I don't have my ID.", "TEST 3: unknown -> verify")
    print("\nDifferent requests, different routes — controlled, traceable, explainable.")







'''
1. Capability given to the agent?  
Creating and logging priority support tickets with different priorities for customer issues.

2. What tool and when is it used?  
`create_priority_ticket` tool, used whenever an issue needs to be tracked and assigned for follow-up.

3. What state fields are tracked?  
`user_message`, `customer_id`, `customer_tier`, `issue_category`, `route_taken`, `ticket_id`, `final_response`.

4. What routing decision does the graph make?  
It routes to `verify_path`, `technical_path`, `premium_path`, or `standard_path` based on `customer_tier` and `issue_category`.

5. Why is this more controlled than a simple agent?  
The flow is deterministic, explainable, and constrained by explicit routing logic instead of open-ended LLM behavior.

'''