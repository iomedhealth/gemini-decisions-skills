# Gemini Decisions API Patterns & Cookbooks

Runnable Python (`google-genai >= 3.2.0`) and TypeScript (`@google/genai >= 3.2.0`) recipes for
building reliable systems with `gemini-decision-1` (`POST /v1beta/decisions`).

All examples assume `GEMINI_API_KEY` is set in the environment and `api_version="v1beta"` /
`apiVersion: "v1beta"` is enabled on the client.

---

## 1. Companion `noul` + `choice` Scope Gate

Because `choice` always normalizes probabilities across the listed options (`sum == 1`), pair it
with a `noul` scope check in the same request rather than relying on a `"none_of_the_above"` option
(alphabetical UTF-8 sorting places `"none_of_the_above"` in the middle of the key order).

### Python

```python
import json
from google import genai

client = genai.Client(http_options={"api_version": "v1beta"})


def route_support_message(message: str) -> str:
    payload = json.dumps({"message": message})
    res = client.decisions.create(
        model="gemini-decision-1",
        input=payload,
        questions={
            "is_support_request": {
                "type": "noul",
                "instructions": "Is `message` an actionable customer support request about our product or billing?",
                "criteria": {
                    "true_description": "Customer reports an issue, asks for account/billing help, or requests a refund/replacement.",
                    "false_description": "Spam, sales pitch, job application, partnership inquiry, or off-topic chatter.",
                },
            },
            "department": {
                "type": "choice",
                "instructions": "Which support department should handle `message`?",
                "criteria": {
                    "account_access": "Login, password reset, MFA, or permissions.",
                    "billing": "Invoices, duplicate charges, refunds, or plan changes.",
                    "shipping": "Delivery status, tracking, or damaged/missing parcels.",
                    "technical": "Bugs, crashes, or product functionality problems.",
                },
            },
        },
    )

    in_scope = res.answers["is_support_request"]
    dept = res.answers["department"]

    if in_scope.noul < 0.30:
        return "ignore_or_archive"
    if in_scope.noul < 0.70 or dept.confidence < 0.75:
        return "human_triage"
    return dept.choice
```

### TypeScript

```typescript
import { GoogleGenAI } from "@google/genai";

const ai = new GoogleGenAI({ apiVersion: "v1beta" });

export async function routeSupportMessage(message: string): Promise<string> {
  const res = await ai.decisions.create({
    model: "gemini-decision-1",
    input: JSON.stringify({ message }),
    questions: {
      is_support_request: {
        type: "noul",
        instructions:
          "Is `message` an actionable customer support request about our product or billing?",
        criteria: {
          true_description:
            "Customer reports an issue, asks for account/billing help, or requests a refund/replacement.",
          false_description:
            "Spam, sales pitch, job application, partnership inquiry, or off-topic chatter.",
        },
      },
      department: {
        type: "choice",
        instructions: "Which support department should handle `message`?",
        criteria: {
          account_access: "Login, password reset, MFA, or permissions.",
          billing: "Invoices, duplicate charges, refunds, or plan changes.",
          shipping: "Delivery status, tracking, or damaged/missing parcels.",
          technical: "Bugs, crashes, or product functionality problems.",
        },
      },
    },
  });

  const inScope = res.answers?.is_support_request;
  const dept = res.answers?.department;
  if (inScope?.type !== "noul" || dept?.type !== "choice") {
    throw new Error("Unexpected missing answers from Decisions API");
  }

  if (inScope.noul < 0.3) return "ignore_or_archive";
  if (inScope.noul < 0.7 || dept.confidence < 0.75) return "human_triage";
  return dept.choice;
}
```

---

## 2. Speculative Fan-Out & Truth-Table Routing

Batch every judgment you might need about a ticket in one request. Shared `input` tokens are billed
once, questions execute in parallel, and your code resolves the truth table — keeping business rules
(like price caps and return windows) deterministic.

### Python

```python
import json
from dataclasses import dataclass
from google import genai

client = genai.Client(http_options={"api_version": "v1beta"})


@dataclass(frozen=True)
class Ticket:
    order_id: str
    item: str
    price_usd: float
    days_since_delivery: int
    tier: str
    message: str


def decide_ticket_action(ticket: Ticket) -> dict:
    # Compute numeric / date policy facts in code before calling the model.
    state = {
        "order": {
            "id": ticket.order_id,
            "item": ticket.item,
            "within_30d_return_window": ticket.days_since_delivery <= 30,
        },
        "customer_tier": ticket.tier,
        "message": ticket.message,
    }

    res = client.decisions.create(
        model="gemini-decision-1",
        input=json.dumps(state),
        questions={
            "is_defect": {
                "type": "noul",
                "instructions": "Does `message` report that `order.item` is broken, damaged, or malfunctioning?",
                "criteria": {
                    "true_description": "Physical damage or functional hardware/software fault.",
                    "false_description": "Preference, sizing, buyer's remorse, or general question.",
                },
            },
            "wants_refund": {
                "type": "noul",
                "instructions": "Is the customer asking for a refund or reversal of a charge?",
            },
            # Speculative: only read if is_defect is high.
            "fault_category": {
                "type": "choice",
                "instructions": "Which defect category best describes the issue in `message`?",
                "criteria": {
                    "cosmetic": "Scratch, dent, or appearance issue that does not block use.",
                    "functional": "Component fails to work as intended.",
                    "missing_part": "Item arrived incomplete or missing accessories.",
                    "other": "Defect outside the above categories.",
                },
            },
            "churn_risk": {
                "type": "score",
                "instructions": "How strongly does `message` signal that the customer may cancel or churn?",
                "criteria": [
                    "Calm inquiry with no churn signal.",
                    "Mild frustration or inconvenience.",
                    "Mentions switching to a competitor.",
                    "Explicitly threatens to cancel or close account.",
                ],
            },
        },
    )

    a = res.answers
    is_defect = a["is_defect"].noul
    wants_refund = a["wants_refund"].noul
    churn = a["churn_risk"]

    # Truth table lives in code, where thresholds and price limits are auditable.
    if churn.score >= 2.2 or ticket.price_usd > 150:
        action = "priority_human_escalation"
    elif is_defect >= 0.80 and wants_refund >= 0.70 and ticket.days_since_delivery <= 30:
        action = "auto_refund"
    elif is_defect >= 0.80 and a["fault_category"].confidence >= 0.75:
        action = f"replacement:{a['fault_category'].choice}"
    elif 0.30 < is_defect < 0.80:
        action = "human_review"
    else:
        action = "self_service_troubleshooting"

    return {
        "action": action,
        "is_defect": is_defect,
        "churn_score": churn.score,
        "churn_probabilities": churn.probabilities,
    }
```

---

## 3. Select Instead of Generate (Pre-Parsed Span & Tool Argument Selection)

Instead of asking a generative model to extract and re-emit strings (which risks typos or
hallucinations), extract exact candidates deterministically in code (regex, parser, or catalog
lookup) and ask `gemini-decision-1` to pick the intended candidate via `choice` + companion `noul`.

### Python

```python
import json
import re
from google import genai

client = genai.Client(http_options={"api_version": "v1beta"})
AMOUNT_RE = re.compile(r"\$\d+(?:,\d{3})*(?:\.\d{2})?")


def extract_refund_amount(email_text: str) -> str | None:
    # 1. Deterministic candidate discovery in code
    candidates = list(dict.fromkeys(AMOUNT_RE.findall(email_text)))
    if not candidates:
        return None

    candidate_map = {f"amt_{i}": val for i, val in enumerate(candidates)}
    state = {"email": email_text, "amount_candidates": candidate_map}

    # 2. If only one candidate exists, a single noul suffices; otherwise noul + choice
    questions: dict = {
        "has_refund_amount": {
            "type": "noul",
            "instructions": "Does `email` state a specific dollar amount that the sender wants refunded?",
        },
    }
    if len(candidate_map) >= 2:
        questions["refund_amount_id"] = {
            "type": "choice",
            "instructions": "Which entry in `amount_candidates` is the exact amount requested for refund?",
            "criteria": {k: f"The dollar amount {v}" for k, v in candidate_map.items()},
        }

    res = client.decisions.create(
        model="gemini-decision-1",
        input=json.dumps(state),
        questions=questions,
    )

    if res.answers["has_refund_amount"].noul < 0.75:
        return None
    if len(candidate_map) == 1:
        return candidates[0]

    pick = res.answers["refund_amount_id"]
    if pick.confidence < 0.75:
        return None  # Escalate to human review
    return candidate_map[pick.choice]
```

---

## 4. RAG Passage Filtering & Citation Verification Cascade

Use `gemini-decision-1` as a fast, zero-output-token filter/verifier over retrieved passages or
draft LLM answers, and escalate only ambiguous or failing cases to a reasoning model on the
Interactions API (`gemini-4-hydrogen` / `gemini-4-carbon`).

### Python

```python
import json
from google import genai

client = genai.Client(http_options={"api_version": "v1beta"})


def verify_grounded_answer(source_passages: list[str], draft_answer: str) -> str:
    # Separate text blocks explicitly with "\n" so content never concatenates mid-word.
    input_blocks = [
        {"type": "text", "text": json.dumps({"sources": source_passages}) + "\n"},
        {"type": "text", "text": json.dumps({"draft_answer": draft_answer})},
    ]

    res = client.decisions.create(
        model="gemini-decision-1",
        input=input_blocks,
        questions={
            "all_claims_supported": {
                "type": "noul",
                "instructions": "Is every factual claim in `draft_answer` directly supported by `sources`?",
                "criteria": {
                    "true_description": "Every factual statement in `draft_answer` is entailed by `sources`.",
                    "false_description": "Contains unsupported extrapolations, invented numbers, or claims absent from `sources`.",
                },
            },
            "contradicts_source": {
                "type": "noul",
                "instructions": "Does `draft_answer` directly contradict any fact in `sources`?",
            },
            "answer_completeness": {
                "type": "score",
                "instructions": "How completely does `draft_answer` reflect the key caveats in `sources`?",
                "criteria": [
                    "Omits critical caveats or misrepresents scope.",
                    "Covers the main point but drops secondary qualifiers.",
                    "Accurately preserves all material caveats and constraints.",
                ],
            },
        },
    )

    supported = res.answers["all_claims_supported"].noul
    contradicts = res.answers["contradicts_source"].noul
    completeness = res.answers["answer_completeness"]

    if contradicts >= 0.40 or supported <= 0.30:
        return "reject_and_regenerate"
    if supported >= 0.85 and completeness.score >= 1.5 and completeness.confidence >= 0.70:
        return "publish"
    return "escalate_to_reasoning_model"
```

---

## 5. Composite Rubric Scoring & Active-Learning Escalation

Score independent quality or risk dimensions once, combine them with transparent weights in code,
and route uncertain or bimodal items to a review queue for active learning.

### TypeScript

```typescript
import { GoogleGenAI } from "@google/genai";

const ai = new GoogleGenAI({ apiVersion: "v1beta" });

export interface ReviewDecision {
  compositeScore: number; // normalized [0, 1]
  route: "auto_approve" | "human_review" | "auto_reject";
  bimodalWarning: boolean;
}

export async function scoreDraftReply(
  customerMessage: string,
  draftReply: string,
): Promise<ReviewDecision> {
  const res = await ai.decisions.create({
    model: "gemini-decision-1",
    input: [
      { type: "text", text: JSON.stringify({ customer_message: customerMessage }) + "\n" },
      { type: "text", text: JSON.stringify({ draft_reply: draftReply }) },
    ],
    questions: {
      clarity: {
        type: "score",
        instructions: "Rate the clarity and actionability of `draft_reply`.",
        criteria: [
          "Confusing or leaves key questions unanswered.",
          "Understandable, but missing concrete next steps or timelines.",
          "Clear, concise, with explicit next steps and timelines.",
        ],
      },
      tone: {
        type: "score",
        instructions: "Rate the professionalism and empathy of `draft_reply`.",
        criteria: [
          "Rude, defensive, or dismissive.",
          "Neutral and polite.",
          "Warm, empathetic, and constructive.",
        ],
      },
      policy_violation: {
        type: "noul",
        instructions: "Does `draft_reply` promise an unauthorized refund, legal commitment, or share another customer's data?",
      },
    },
  });

  const clarity = res.answers?.clarity;
  const tone = res.answers?.tone;
  const violation = res.answers?.policy_violation;
  if (clarity?.type !== "score" || tone?.type !== "score" || violation?.type !== "noul") {
    throw new Error("Incomplete answers from Decisions API");
  }

  // Hard safety gate first
  if (violation.noul >= 0.35) {
    return { compositeScore: 0, route: "auto_reject", bimodalWarning: false };
  }

  // Normalize 0..2 scores onto [0, 1] and apply business weights in code
  const clarityNorm = clarity.score / 2;
  const toneNorm = tone.score / 2;
  const compositeScore = 0.65 * clarityNorm + 0.35 * toneNorm;

  // Detect bimodal split (probability mass at both extremes 0 and 2)
  const bimodalWarning =
    ((clarity.probabilities["0"] ?? 0) > 0.25 && (clarity.probabilities["2"] ?? 0) > 0.25) ||
    ((tone.probabilities["0"] ?? 0) > 0.25 && (tone.probabilities["2"] ?? 0) > 0.25);

  if (bimodalWarning || clarity.confidence < 0.7 || tone.confidence < 0.7) {
    return { compositeScore, route: "human_review", bimodalWarning };
  }
  if (compositeScore >= 0.8) {
    return { compositeScore, route: "auto_approve", bimodalWarning: false };
  }
  return { compositeScore, route: "human_review", bimodalWarning: false };
}
```

---

## 6. Multimodal Triage (PDF, Audio, Video)

All media lives in the shared `input` array; every question runs over that media in one pass with
zero output tokens. Use inline base64 for smaller files (`<= 50 MB` PDFs, `<= 100 MB` total request),
public/signed HTTPS URLs up to `15 MB`, or `client.files.upload(...)` for larger/reused assets. Do
not pass `gs://` or YouTube URLs.

### A. PDF Invoice Triage (Python)

```python
import base64
from google import genai

client = genai.Client(http_options={"api_version": "v1beta"})


def triage_invoice_pdf(pdf_path: str) -> dict:
    with open(pdf_path, "rb") as f:
        pdf_b64 = base64.b64encode(f.read()).decode("utf-8")

    res = client.decisions.create(
        model="gemini-decision-1",
        input=[
            {
                "type": "document",
                "data": pdf_b64,
                "mime_type": "application/pdf",
            },
        ],
        questions={
            "has_po_number": {
                "type": "noul",
                "instructions": "Does the invoice display a purchase order (PO) number?",
            },
            "is_past_due": {
                "type": "noul",
                "instructions": "Does the invoice state that payment is overdue or past due?",
            },
            "expense_category": {
                "type": "choice",
                "instructions": "Which expense category best fits this invoice?",
                "criteria": {
                    "cloud_infrastructure": "Servers, GPU compute, hosting, or storage.",
                    "professional_services": "Consulting, legal, accounting, or contractor fees.",
                    "software_licenses": "SaaS subscriptions or seat licenses.",
                    "travel": "Flights, hotels, meals, or ground transport.",
                    "other": "Anything outside these categories.",
                },
            },
        },
    )

    po = res.answers["has_po_number"]
    cat = res.answers["expense_category"]
    return {
        "has_po_number": po.noul >= 0.75,
        "po_probability": po.noul,
        "category": cat.choice if cat.confidence >= 0.75 else "manual_coding",
        "category_confidence": cat.confidence,
    }
```

### B. Audio Call QA Without Transcription & Low-Res Video Screening via Files API (Python)

```python
from google import genai

client = genai.Client(http_options={"api_version": "v1beta"})


def audit_support_call(audio_path: str) -> dict:
    # Upload once for local files > 15 MB or files reused across calls.
    uploaded = client.files.upload(file=audio_path)

    res = client.decisions.create(
        model="gemini-decision-1",
        input=[
            {
                "type": "audio",
                "uri": uploaded.uri,
                "mime_type": uploaded.mime_type,
            },
        ],
        questions={
            "caller_frustrated": {
                "type": "noul",
                "instructions": "Does the caller sound frustrated, angry, or impatient based on words and vocal tone?",
            },
            "agent_clarity": {
                "type": "score",
                "instructions": "How clearly and calmly did the support agent explain the resolution?",
                "criteria": [
                    "Unclear, hesitant, or talked over the caller.",
                    "Understandable with minor pauses or repetition.",
                    "Clear, calm, and well-structured throughout.",
                ],
            },
        },
    )
    return {
        "caller_frustrated": res.answers["caller_frustrated"].noul,
        "agent_clarity": res.answers["agent_clarity"].score,
    }


def screen_clip_for_safety_hazard(video_path: str) -> float:
    uploaded = client.files.upload(file=video_path)
    res = client.decisions.create(
        model="gemini-decision-1",
        input=[
            {
                "type": "video",
                "uri": uploaded.uri,
                "mime_type": uploaded.mime_type,
                "resolution": "low",  # Use "low" when fine visual detail is unnecessary
            },
        ],
        questions={
            "shows_fire_or_smoke": {
                "type": "noul",
                "instructions": "Does the video show visible fire, flames, or heavy smoke?",
            },
        },
    )
    return res.answers["shows_fire_or_smoke"].noul
```
