
# smart_support_bot.py
# Build a Smart Customer Support Bot using OpenAI Agent SDK
# Run: python smart_support_bot.py
#
# Demonstrates:
# - Two agents (BotAgent, HumanAgent)
# - @function_tool with is_enabled and error_function
# - @guardrail to filter negative/offensive input
# - Handoff from BotAgent -> HumanAgent
# - model_settings (tool_choice, metadata)
# - Logging of tool invocations and handoffs

from dataclasses import dataclass
import logging
import re
from typing import Dict, Any

from openai import OpenAI
from openai.agents import Agent, Guardrail, Handoff
from openai.agents.decorators import function_tool, guardrail

# -----------------------------------------------------------------------------
# Setup
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

client = OpenAI()

# -----------------------------------------------------------------------------
# Simulated datastore (Orders + FAQs)
# -----------------------------------------------------------------------------
ORDERS = {
    "ORD-1001": {"status": "Shipped", "eta": "2-3 business days", "carrier": "DHL"},
    "ORD-1002": {"status": "Processing", "eta": "N/A", "carrier": None},
    "ORD-1003": {"status": "Delivered", "eta": "N/A", "carrier": "FedEx"},
}

FAQS = {
    "return policy": "You can return unopened items within 30 days for a full refund.",
    "shipping time": "Standard shipping takes 3–5 business days. Expedited options are available.",
    "warranty": "All electronics include a 1-year limited warranty covering manufacturing defects."
}

OFFENSIVE_PATTERNS = [
    r"\b(dumb|stupid|idiot|shut up|hate|useless)\b",
    r"\b(f\*?ck|s\*?it|b\*?tch|a\*?shole)\b"
]

NEGATIVE_SENTIMENT_PATTERNS = [
    r"\b(angry|furious|terrible|awful|worst|hate|refund now)\b",
    r"\b(complain|complaint|speak to manager|not happy|disappointed)\b"
]

# -----------------------------------------------------------------------------
# Helper: intent & sentiment detection (very simple heuristics for demo)
# -----------------------------------------------------------------------------
def is_order_intent(text: str) -> bool:
    text = text.lower()
    return "order" in text or "track" in text or "status" in text

def extract_order_id(text: str) -> str | None:
    # Look for tokens like ORD-1234 or ORD1234
    m = re.search(r"\bORD[- ]?(\d{3,6})\b", text.upper())
    if m:
        num = m.group(1)
        return f"ORD-{num}"
    return None

def is_negative_or_offensive(text: str) -> bool:
    t = text.lower()
    for pat in OFFENSIVE_PATTERNS + NEGATIVE_SENTIMENT_PATTERNS:
        if re.search(pat, t):
            return True
    return False

def match_faq(text: str) -> str | None:
    t = text.lower()
    for k, v in FAQS.items():
        if k in t:
            return v
    return None

# -----------------------------------------------------------------------------
# Guardrail: Filter offensive/negative input and gently steer tone
# -----------------------------------------------------------------------------
@guardrail(
    name="civility_guardrail",
    description="Blocks or reframes negative/offensive language to keep interactions positive.",
    type="input"
)
def civility_guardrail_fn(user_input: str):
    if is_negative_or_offensive(user_input):
        # Return False + replacement text to stop raw input and reframe
        gentle = (
            "I’m here to help. Let’s keep things respectful so I can assist you quickly. "
            "Could you please restate your request? For example: "
            "‘Please check my order status ORD-1234’ or ‘What is your return policy?’"
        )
        return False, gentle
    # Allow input to pass through
    return True, user_input

civility_guardrail = Guardrail(
    name="civility_guardrail",
    description="Keeps interactions positive and respectful.",
    fn=civility_guardrail_fn,
    type="input"
)

# -----------------------------------------------------------------------------
# Function Tool: get_order_status with is_enabled + error_function
# -----------------------------------------------------------------------------
def _order_tool_enabled(context: Dict[str, Any]) -> bool:
    # Enable only when user's latest message has order intent
    text = context.get("last_user_message", "")
    enabled = is_order_intent(text)
    logging.info(f"[Tool Toggle] get_order_status enabled? {enabled} (message='{text}')")
    return enabled

def _order_tool_error(err: Exception, args: Dict[str, Any]):
    # Friendly error message returned to the user
    order_id = args.get("order_id", "UNKNOWN")
    msg = (
        f"Sorry, I couldn’t find details for order '{order_id}'. "
        "Please double-check the ID (e.g., ORD-1001) or ask me to connect you to a human agent."
    )
    logging.error(f"[Tool Error] get_order_status: {err} (order_id={order_id})")
    return msg

@function_tool(
    name="get_order_status",
    description="Fetches simulated order status data by order_id.",
    is_enabled=_order_tool_enabled,
    error_function=_order_tool_error
)
def get_order_status(order_id: str):
    logging.info(f"[Tool Invoke] get_order_status(order_id={order_id})")
    if order_id not in ORDERS:
        # Raising triggers error_function to return a friendly message
        raise KeyError(f"Order not found: {order_id}")
    data = ORDERS[order_id]
    parts = [f"Status: {data['status']}"]
    if data["eta"] and data["eta"] != "N/A":
        parts.append(f"ETA: {data['eta']}")
    if data["carrier"]:
        parts.append(f"Carrier: {data['carrier']}")
    return f"Order {order_id} → " + ", ".join(parts)

# -----------------------------------------------------------------------------
# Agents
# -----------------------------------------------------------------------------
@dataclass
class ModelSettings:
    tool_choice: str = "auto"   # "auto" or "required"
    metadata: Dict[str, Any] = None

# Human agent: receives handoffs
human_agent = Agent(
    client=client,
    model="gpt-4.1-mini",
    instructions=(
        "You are a human support specialist. "
        "Be empathetic, clarify the issue, and resolve or collect details for escalation tickets."
    )
)

# Bot agent: answers FAQs, looks up orders, and may escalate
bot_agent = Agent(
    client=client,
    model="gpt-4.1-mini",
    tools=[get_order_status],
    guardrails=[civility_guardrail],
    instructions=(
        "You are a helpful Customer Support Bot. "
        "1) Answer common FAQs succinctly. "
        "2) For order inquiries, use the get_order_status tool if available. "
        "3) If the query is complex or sentiment is negative, handoff to the HumanAgent. "
        "Always be polite, concise, and solution-focused."
    )
)

# -----------------------------------------------------------------------------
# Routing / Orchestration
# -----------------------------------------------------------------------------
def handle_user_message(
    message: str,
    customer_id: str,
    model_settings: ModelSettings = ModelSettings(tool_choice="auto", metadata=None)
):
    """
    Main entry point that:
      - Applies guardrails (handled by Agent automatically via guardrails list)
      - Uses FAQs or tools as needed
      - Performs handoff when appropriate
      - Demonstrates model_settings (tool_choice + metadata)
    """

    # 1) Quick sentiment/complexity check for handoff signals
    negative = is_negative_or_offensive(message)
    order_intent = is_order_intent(message)
    order_id = extract_order_id(message)

    # 2) Decide if we should immediately escalate (very negative input)
    if negative:
        logging.info("[Decision] Negative sentiment detected → Handoff to HumanAgent")
        return Handoff(
            to=human_agent,
            reason="Detected negative sentiment or offensive language. Human empathy required.",
            context={"last_user_message": message, "customer_id": customer_id}
        )

    # 3) Try FAQs first
    faq_answer = match_faq(message)
    if faq_answer and not order_intent:
        logging.info("[Decision] Answering FAQ")
        return faq_answer

    # 4) Order tool path (respect tool_choice)
    ctx = {"last_user_message": message, "customer_id": customer_id}
    response_metadata = model_settings.metadata or {}
    response_metadata["customer_id"] = customer_id

    if order_intent:
        if model_settings.tool_choice == "required":
            # Force tool path if enabled; otherwise escalate
            if _order_tool_enabled(ctx) and order_id:
                return bot_agent.run(
                    f"Fetch the status for {order_id}.",
                    context=ctx,
                    model_settings={"tool_choice": "required", "metadata": response_metadata}
                )
            else:
                logging.info("[Decision] Tool required but not usable → Handoff")
                return Handoff(
                    to=human_agent,
                    reason="Order tool not available or order ID missing; requires human assistance.",
                    context={"last_user_message": message, "customer_id": customer_id}
                )
        else:
            # tool_choice="auto" → Bot decides
            if order_id:
                return bot_agent.run(
                    f"User asks to track order {order_id}. Use the tool if available.",
                    context=ctx,
                    model_settings={"tool_choice": "auto", "metadata": response_metadata}
                )
            else:
                # Ask the bot to collect the ID or escalate
                logging.info("[Decision] Order intent without ID → Ask for ID (bot) or Handoff if user insists)")
                return "Could you share your order ID (e.g., ORD-1001) so I can check the status?"

    # 5) If not FAQ and not order: simple capability or escalate for complexity
    # Heuristic: if message is long and ambiguous → escalate.
    if len(message.split()) > 40 or "complicated" in message.lower() or "legal" in message.lower():
        logging.info("[Decision] Complex query → Handoff to HumanAgent")
        return Handoff(
            to=human_agent,
            reason="Complex or ambiguous request.",
            context={"last_user_message": message, "customer_id": customer_id}
        )

    # Default helpful response
    logging.info("[Decision] Default helpful response")
    return (
        "I can help with order tracking (share your order ID like ORD-1001) "
        "or answer policies like shipping time, warranty, and returns."
    )

# -----------------------------------------------------------------------------
# Demo / Examples
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    print("\n--- Demo 1: Friendly FAQ ---")
    out1 = handle_user_message(
        "Hi! What’s your return policy?",
        customer_id="CUST-789"
    )
    print(out1 if not isinstance(out1, Handoff) else f"[HANDOFF] {out1.reason}")

    print("\n--- Demo 2: Order tracking with ID (tool auto) ---")
    out2 = handle_user_message(
        "Can you track my order ORD-1001?",
        customer_id="CUST-789",
        model_settings=ModelSettings(tool_choice="auto", metadata={"channel": "web"})
    )
    if isinstance(out2, Handoff):
        print(f"[HANDOFF] {out2.reason}")
    else:
        print(out2)

    print("\n--- Demo 3: Order tool REQUIRED, valid ID ---")
    out3 = handle_user_message(
        "Status for ORD-1002 please.",
        customer_id="CUST-001",
        model_settings=ModelSettings(tool_choice="required", metadata={"channel": "mobile-app"})
    )
    print(out3 if not isinstance(out3, Handoff) else f"[HANDOFF] {out3.reason}")

    print("\n--- Demo 4: Missing ID (tool auto) ---")
    out4 = handle_user_message(
        "I want the status of my order",
        customer_id="CUST-002",
        model_settings=ModelSettings(tool_choice="auto", metadata={"channel": "web"})
    )
    print(out4 if not isinstance(out4, Handoff) else f"[HANDOFF] {out4.reason}")

    print("\n--- Demo 5: Unknown order (error_function path) ---")
    out5 = handle_user_message(
        "Track ORD-9999 for me",
        customer_id="CUST-002",
        model_settings=ModelSettings(tool_choice="auto", metadata={"channel": "web"})
    )
    print(out5 if not isinstance(out5, Handoff) else f"[HANDOFF] {out5.reason}")

    print("\n--- Demo 6: Offensive/negative message triggers guardrail & escalation ---")
    out6 = handle_user_message(
        "This is the worst service ever, I’m furious! Track my damn order ORD-1003.",
        customer_id="CUST-123",
        model_settings=ModelSettings(tool_choice="auto", metadata={"channel": "web"})
    )
    if isinstance(out6, Handoff):
        print(f"[HANDOFF] {out6.reason}")
    else:
        print(out6)

    print("\n--- Demo 7: Complex/ambiguous → Handoff ---")
    long_msg = "Hello, I need help with a multi-country warranty claim involving two shipments, a customs dispute, and a missed delivery window, plus legal follow-up."
    out7 = handle_user_message(
        long_msg,
        customer_id="CUST-555",
        model_settings=ModelSettings(tool_choice="auto", metadata={"channel": "email"})
    )
    print(out7 if not isinstance(out7, Handoff) else f"[HANDOFF] {out7.reason}")
