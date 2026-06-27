"""Pydantic schemas for mira_math v2 agent messages.

Key differences from v1:
- Flexible natural language requests (no hardcoded hint kinds)
- Agent B can decline if request doesn't match what B has
- Designed for multi-turn retry loops

FIXES:
- Replaced AgentBMsgEnvelope (nested Union) with AgentBFlatResponse (flat schema).
  Root cause: with_structured_output(method="function_calling") emits a JSON Schema.
  When the target field is `message: Union[OfferMsg, DeclineMsg]` the LLM receives a
  complex `anyOf` schema and frequently fills `message` with a plain string instead
  of a nested dict, causing a Pydantic validation error on every call and making
  Agent B always appear to decline even when it actually has the answer.
  A flat schema with a `type` discriminator is far more reliably followed.

- Removed FinalMsgEnvelope for the same reason: FinalMsg is already flat (type,
  reasoning, answer, consistency_check) so using it directly with
  with_structured_output avoids the nested-message problem entirely.

- Removed available_hint from DeclineMsg and AgentBFlatResponse.
  Agent B should only: (a) provide information if it has it, or (b) say it does not
  have that specific information. No partial hints.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Literal, Optional, Union
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

JSONValue = Union[None, bool, int, float, str, List[Any], Dict[str, Any]]


def _clip_str(v: Any, n: int) -> str:
    if v is None:
        return ""
    s = str(v)
    return s if len(s) <= n else s[:n]


# ---------------------------------------------------------------------------
# Shared sub-models
# ---------------------------------------------------------------------------

class Progress(BaseModel):
    """Captures intermediate reasoning state."""
    model_config = ConfigDict(extra="ignore")

    solved_vars: Optional[List[str]] = Field(
        default=None,
        description="Variables determined uniquely from current constraints.",
    )
    free_vars: Optional[List[str]] = Field(
        default=None,
        description="Variables that remain undetermined (degrees of freedom).",
    )
    expressions: Optional[Dict[str, str]] = Field(
        default=None,
        description="Parametric solutions: variable -> expression in terms of free vars.",
    )
    candidates: Optional[List[Dict[str, JSONValue]]] = Field(
        default=None,
        description="Finite candidate solutions consistent with current view.",
        max_length=20,
    )
    notes: Optional[str] = Field(
        default=None,
        description="Brief explanation of current reasoning state (1-2 sentences).",
        max_length=300,
    )

    @field_validator("notes", mode="before")
    def _clip_notes(cls, v: Any) -> Any:
        if v is None:
            return None
        return _clip_str(v, 300)


# ---------------------------------------------------------------------------
# Agent A messages
# ---------------------------------------------------------------------------

class RequestMsg(BaseModel):
    """Agent A requests missing information from Agent B."""
    model_config = ConfigDict(extra="ignore")

    type: Literal["request"] = Field(
        default="request",
        description="Message type indicating a request for information.",
    )
    reasoning: str = Field(
        ...,
        description=(
            "Your analysis of the problem: what constraints you have, what you can solve, "
            "and what specific information is missing to reach a unique solution."
        ),
        max_length=500,
    )
    request: str = Field(
        ...,
        description=(
            "A precise natural language request for the ONE piece of information you need. "
            "Be specific: e.g., 'the value of variable c', 'the sign of x0 (+1 or -1)', "
            "'the point p(3)', 'the constraint involving z'. The other agent will only "
            "provide information if your request exactly matches what they have."
        ),
        max_length=200,
    )
    progress: Progress = Field(
        default_factory=Progress,
        description="Your current solution progress showing what you've solved so far.",
    )

    @field_validator("reasoning", mode="before")
    def _clip_reasoning(cls, v: Any) -> str:
        return _clip_str(v, 500)

    @field_validator("request", mode="before")
    def _clip_request(cls, v: Any) -> str:
        return _clip_str(v, 200)

    @field_validator("progress", mode="before")
    def _coerce_progress(cls, v: Any) -> Any:
        return {} if v is None else v


class FinalMsg(BaseModel):
    """Agent A provides the final answer.

    Used DIRECTLY with with_structured_output (no envelope wrapper needed — the
    fields are already at top level, so there is no nested-message parsing risk).
    """
    model_config = ConfigDict(extra="ignore")

    type: Literal["final"] = Field(
        default="final",
        description="Message type for the final answer.",
    )
    reasoning: str = Field(
        ...,
        description=(
            "Explain step-by-step how you solved the problem using your original constraints "
            "combined with the new information from Agent B. Show your work."
        ),
    )
    answer: JSONValue = Field(
        ...,
        description="The final answer in the required format for this problem type.",
    )
    consistency_check: Optional[str] = Field(
        default=None,
        description="Brief verification that the answer satisfies all constraints.",
    )


class FinalMsgGemini(BaseModel):
    """Gemini-compatible variant of FinalMsg.

    Avoids JSONValue (Union/anyOf) and Literal (enum) which Gemini's function
    calling schema does not support via $defs.  Uses Dict[str, Any] for answer
    so Gemini can populate {"numerator": N, "denominator": D}, {"value": N},
    or {"var": N, ...} directly.
    """
    model_config = ConfigDict(extra="ignore")

    type: str = Field(
        default="final",
        description="Always set this to 'final'.",
    )
    reasoning: str = Field(
        ...,
        description=(
            "Explain step-by-step how you solved the problem using your original constraints "
            "combined with the new information from Agent B. Show your work."
        ),
    )
    answer: Dict[str, Any] = Field(
        ...,
        description=(
            "The final answer as a JSON object. "
            "For a single integer answer use {\"value\": N}. "
            "For a fraction use {\"numerator\": N, \"denominator\": D} in lowest terms. "
            "For multiple variables use {\"var1\": val1, \"var2\": val2, ...}."
        ),
    )
    consistency_check: str = Field(
        default="",
        description="Brief verification that the answer satisfies all constraints.",
    )


# ---------------------------------------------------------------------------
# Agent B messages
# ---------------------------------------------------------------------------

class OfferMsg(BaseModel):
    """Agent B provides requested information."""
    model_config = ConfigDict(extra="ignore")

    type: Literal["offer"] = Field(
        default="offer",
        description="Message type indicating an offer of information.",
    )
    has_exact_match: bool = Field(
        ...,
        description=(
            "MUST be True ONLY if Agent A's request exactly matches information you have "
            "in your private constraints. If False, you should use 'decline' instead."
        ),
    )
    constraint_quoted: str = Field(
        ...,
        description=(
            "Copy-paste the EXACT constraint from your private data that matches the request. "
            "Do NOT make up information - only quote what you actually have."
        ),
        max_length=300,
    )
    hint: str = Field(
        ...,
        description=(
            "The specific information extracted from your quoted constraint. "
            "State it clearly: e.g., 'c = 6', 'sign(x0) = +1', 'p(3) = 15'."
        ),
        max_length=200,
    )

    @field_validator("constraint_quoted", mode="before")
    def _clip_constraint(cls, v: Any) -> str:
        return _clip_str(v, 300)

    @field_validator("hint", mode="before")
    def _clip_hint(cls, v: Any) -> str:
        return _clip_str(v, 200)


class DeclineMsg(BaseModel):
    """Agent B declines — does not have what was requested.

    Note: available_hint has been intentionally removed. Agent B should only
    confirm it has the requested information (offer) or state that it does not
    (decline). No partial hints are provided.
    """
    model_config = ConfigDict(extra="ignore")

    type: Literal["decline"] = Field(
        default="decline",
        description="Message type indicating the requested information is not available.",
    )
    message: str = Field(
        default="I do not have that specific information.",
        description="Explanation that the requested info is not available.",
        max_length=200,
    )

    @field_validator("message", mode="before")
    def _clip_message(cls, v: Any) -> str:
        return _clip_str(v, 200)


# ---------------------------------------------------------------------------
# FLAT response schema for Agent B
# ---------------------------------------------------------------------------
# WHY THIS EXISTS:
# The original AgentBMsgEnvelope used `message: Union[OfferMsg, DeclineMsg]`.
# When converted to JSON Schema for function-calling, this becomes a complex
# `anyOf` with deeply nested objects. gpt-4o-mini would routinely place a plain
# text string in `message` rather than the required nested dict; Pydantic rejected
# that, and the fallback always produced a decline — even when B held the answer.
#
# A flat schema with a single `type` discriminator is far more reliably followed.

class AgentBFlatResponse(BaseModel):
    """
    Flat, discriminated schema for Agent B's response.

    Offer (type="offer", has_exact_match=True):
        type              → "offer"
        has_exact_match   → true
        constraint_quoted → exact constraint text copied from private data
        hint              → extracted value, e.g. "a(1) = 7"
        decline_message   → null

    Decline (type="decline"):
        type              → "decline"
        has_exact_match   → false
        constraint_quoted → null
        hint              → null
        decline_message   → brief explanation
    """
    model_config = ConfigDict(extra="ignore")

    type: Literal["offer", "decline"] = Field(
        ...,
        description=(
            "Response type: 'offer' if you have an EXACT match for the request, "
            "'decline' if you do not have that specific information."
        ),
    )
    has_exact_match: bool = Field(
        default=False,
        description=(
            "True ONLY if your private constraints contain exactly what Agent A requested. "
            "Must be True when type='offer', False when type='decline'."
        ),
    )
    constraint_quoted: Optional[str] = Field(
        default=None,
        description=(
            "OFFER ONLY: Copy-paste the EXACT constraint text from your private data. "
            "Leave null when declining."
        ),
        max_length=300,
    )
    hint: Optional[str] = Field(
        default=None,
        description=(
            "OFFER ONLY: The specific value/information extracted from your constraint. "
            "E.g. 'a(1) = 7', 'sign(x0) = +1', 'p(3) = 15'. Leave null when declining."
        ),
        max_length=200,
    )
    decline_message: Optional[str] = Field(
        default=None,
        description=(
            "DECLINE ONLY: Brief explanation that you do not have the requested info. "
            "Leave null when offering."
        ),
        max_length=200,
    )

    @field_validator("constraint_quoted", mode="before")
    def _clip_cq(cls, v: Any) -> Any:
        return None if v is None else _clip_str(v, 300)

    @field_validator("hint", mode="before")
    def _clip_hint(cls, v: Any) -> Any:
        return None if v is None else _clip_str(v, 200)

    @field_validator("decline_message", mode="before")
    def _clip_dm(cls, v: Any) -> Any:
        return None if v is None else _clip_str(v, 200)

    def to_agent_b_msg(self) -> Union[OfferMsg, DeclineMsg]:
        """Convert flat response to the canonical OfferMsg / DeclineMsg."""
        if self.type == "offer" and self.has_exact_match:
            return OfferMsg(
                type="offer",
                has_exact_match=True,
                constraint_quoted=self.constraint_quoted or "",
                hint=self.hint or "",
            )
        return DeclineMsg(
            type="decline",
            message=(
                self.decline_message
                or "I do not have that specific information."
            ),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Return the canonical message as a plain dict (for transcript storage)."""
        return self.to_agent_b_msg().model_dump()


# ---------------------------------------------------------------------------
# Type unions (kept for type hints)
# ---------------------------------------------------------------------------
AgentAMsg = Union[RequestMsg, FinalMsg]
AgentBMsg = Union[OfferMsg, DeclineMsg]
AllMessages = Union[RequestMsg, OfferMsg, DeclineMsg, FinalMsg]


# ---------------------------------------------------------------------------
# Envelope for Agent A's request
# ---------------------------------------------------------------------------

class RequestMsgEnvelope(BaseModel):
    """
    Envelope for Agent A's request.

    Using an envelope keeps the function-calling schema clean: the LLM receives
    a single top-level function argument `message` whose schema is RequestMsg.

    extra="ignore" (not "forbid") because some LLMs emit recognised RequestMsg
    fields (e.g. "progress", "type") at the envelope level in addition to
    inside "message".  We silently drop those top-level duplicates; the correct
    values are already inside "message".
    """
    model_config = ConfigDict(extra="ignore")
    message: RequestMsg = Field(..., description="Agent A's request message.")

    _REQUEST_MSG_FIELDS = {"type", "reasoning", "request", "progress"}

    @model_validator(mode="before")
    def _wrap_unenveloped(cls, v: Any) -> Any:
        """
        Normalise the three layouts the LLM may produce:

        1. Correct envelope  → {"message": {type, reasoning, request, progress}}
           Leave as-is.

        2. Flat / un-enveloped → {type, reasoning, request, progress}
           Wrap into {"message": v}.

        3. Envelope + stray top-level RequestMsg fields →
           {"message": {type, reasoning, request}, "progress": {}, ...}
           Merge the stray fields into "message" if they are absent there,
           then drop them from the top level (extra="ignore" handles the drop).
        """
        if not isinstance(v, dict):
            return v

        _REQUEST_MSG_FIELDS = {"type", "reasoning", "request", "progress"}

        if "message" not in v:
            # Case 2: flat dict — wrap if it looks like a request
            if v.get("type") == "request":
                return {"message": v}
            return v

        # Case 3: message key present but stray RequestMsg fields also at top level
        msg = v.get("message")
        if isinstance(msg, dict):
            for field in _REQUEST_MSG_FIELDS:
                if field in v and field not in msg:
                    msg = dict(msg)  # copy before mutating
                    msg[field] = v[field]
            return {"message": msg}

        return v


class RequestMsgGemini(BaseModel):
    """Gemini-compatible flat request schema.

    Avoids $defs/$ref nesting (RequestMsgEnvelope -> RequestMsg -> Progress)
    which Gemini's function calling drops.  All fields are top-level with
    simple types only.
    """
    model_config = ConfigDict(extra="ignore")

    type: str = Field(
        default="request",
        description="Always set this to 'request'.",
    )
    reasoning: str = Field(
        default="",
        description=(
            "Your analysis of the problem: what constraints you have, what you can solve, "
            "and what specific information is missing to reach a unique solution."
        ),
    )
    request: str = Field(
        ...,
        description=(
            "A precise natural language request for the ONE piece of information you need. "
            "Be specific: e.g., 'the value of variable c', 'the sign of x0 (+1 or -1)', "
            "'the point p(3)', 'the constraint involving z'. The other agent will only "
            "provide information if your request exactly matches what they have."
        ),
    )


# NOTE: FinalMsgEnvelope is intentionally removed.
# Use FinalMsg directly with with_structured_output().
# FinalMsg is already flat (type, reasoning, answer, consistency_check),
# so there is no nested-message parsing risk. The envelope only added the
# same nesting bug that was fixed in AgentBFlatResponse.

# NOTE: AgentBMsgEnvelope is intentionally removed.
# Use AgentBFlatResponse directly.


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_max_requests(difficulty: int) -> int:
    """Get maximum allowed request attempts based on difficulty."""
    limits = {1: 3, 2: 6, 3: 9}
    return limits.get(difficulty, 3)