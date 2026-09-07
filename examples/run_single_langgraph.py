#!/usr/bin/env python3
"""mira_math Runner using LangGraph.

Key features:
- Flexible natural language requests (no hardcoded hint kinds)
- Retry loop: Agent B declines if request doesn't match, A can retry
- Difficulty-based retry limits (3/6/9 for difficulty 1/2/3)
- Modular LangGraph design for extensibility
- Supports multiple agent methods: llm, react, reflexion

FIXES:
- agent_b_respond: replaced AgentBMsgEnvelope (nested Union) with AgentBFlatResponse
  (flat discriminated schema).
- _agent_a_final_impl: replaced FinalMsgEnvelope (nested envelope) with FinalMsg
  used directly. FinalMsg is already flat so no nested-message parse failure.
- datetime.utcnow() → datetime.now(timezone.utc).
- Agent A retry prompt: injects negotiation history so A does not repeat declined
  requests and knows to ask for a different type of information.
- Removed available_hint: Agent B only offers or declines, no partial hints.
- Removed all string truncation from verbose output, logging, and display.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, TypedDict, Literal

from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import create_react_agent

from mira_math.agent_schema import (
    RequestMsgEnvelope,
    RequestMsgGemini,
    AgentBFlatResponse,
    FinalMsg,
    FinalMsgGemini,
    get_max_requests,
)
from mira_math.generate import _format_ideal_request
from mira_math.oracle import (
    build_oracle_instance,
    select_b_resolving_constraint,
    CONDITION_NAME,
)
from mira_math.prompts import (
    system_prompt_agent_a,
    system_prompt_agent_b,
    user_prompt_agent_a,
    user_prompt_agent_b,
)
from mira_math.tools import CALCULATOR_TOOL


# ---------------------------------------------------------------------------
# State Definition
# ---------------------------------------------------------------------------
class MASState(TypedDict):
    """State for the multi-agent system."""
    instance: Dict[str, Any]
    shared_history: List[Dict[str, Any]]    # Messages visible to both agents
    transcript: List[Dict[str, Any]]         # Full transcript with metadata
    request_count: int                       # Number of requests made by A
    max_requests: int                        # Maximum allowed requests
    last_request: Optional[Dict[str, Any]]  # Most recent request from A
    phase: Literal["request", "respond", "final", "failed", "done"]
    got_hint: bool                           # Whether A received useful info
    method: str                              # Agent method: llm, react, reflexion
    tools_used: List[str]                    # Names of tools available
    verbose: bool                            # Show detailed method traces
    declined_requests: List[str]             # Requests B has already declined


# ---------------------------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------------------------
def _load_env(path: str = ".env") -> None:
    """Load environment variables from .env file."""
    candidates = [path]
    if not os.path.isabs(path):
        candidates.append(os.path.join(os.path.dirname(__file__), "..", path))
    for candidate in candidates:
        if os.path.exists(candidate):
            with open(candidate, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    key, value = line.split("=", 1)
                    key = key.strip()
                    value = value.strip().strip("\"'")
                    if key and key not in os.environ:
                        os.environ[key] = value
            break


def _maybe_json_load(value: Any) -> Optional[Dict[str, Any]]:
    """Try to parse JSON from various formats."""
    if value is None:
        return None
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            pass
        text = value.strip()
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                pass
    return None


def _extract_answer_from_reasoning(reasoning: str, logger: logging.Logger) -> Any:
    """Last-resort extraction of an answer embedded in the reasoning text.

    LLMs sometimes put the answer inside the reasoning field instead of the
    answer field.  We look for JSON objects/values near common answer phrases.
    Returns the extracted answer or None.
    """
    if not reasoning:
        return None

    # Look for the last JSON object in the reasoning (likely the answer dict)
    json_objects = []
    for m in re.finditer(r'\{[^{}]+\}', reasoning):
        try:
            obj = json.loads(m.group())
            json_objects.append(obj)
        except json.JSONDecodeError:
            pass

    if json_objects:
        answer = json_objects[-1]
        logger.info("Extracted answer from reasoning text (JSON object): %s", answer)
        return answer

    # Look for a plain integer/number after common answer phrases
    m = re.search(
        r'(?:final answer|answer|result|value)\s*(?:is|=|:)\s*(-?\d+(?:/\d+)?)',
        reasoning,
        re.IGNORECASE,
    )
    if m:
        val = m.group(1)
        if '/' in val:
            num, den = val.split('/')
            answer = {"numerator": int(num), "denominator": int(den)}
        else:
            answer = {"value": int(val)}
        logger.info("Extracted answer from reasoning text (regex): %s", answer)
        return answer

    # Last resort: take the last fraction or integer in the reasoning.
    # Handles cases like "... simplest form: 12/30 = 2/5." where there is
    # no explicit "answer is" prefix.
    fractions = list(re.finditer(r'(-?\d+)/(\d+)', reasoning))
    if fractions:
        last = fractions[-1]
        answer = {"numerator": int(last.group(1)), "denominator": int(last.group(2))}
        logger.info("Extracted answer from reasoning text (last fraction): %s", answer)
        return answer

    integers = list(re.finditer(r'(?<![/\d])(-?\d+)(?![/\d])', reasoning))
    if integers:
        answer = {"value": int(integers[-1].group(1))}
        logger.info("Extracted answer from reasoning text (last integer): %s", answer)
        return answer

    logger.warning(
        "Could not extract answer from final message reasoning. "
        "Reasoning tail: %s", reasoning[-200:] if len(reasoning) > 200 else reasoning,
    )
    return None


def _extract_from_raw(raw: Any, logger: logging.Logger) -> Optional[Dict[str, Any]]:
    """Extract message dict from raw LLM response."""
    if raw is None:
        return None

    # Try tool_calls first (function calling)
    tool_calls = getattr(raw, "tool_calls", None)
    if not tool_calls and hasattr(raw, "additional_kwargs"):
        tool_calls = raw.additional_kwargs.get("tool_calls")

    if tool_calls:
        first = tool_calls[0]
        if hasattr(first, "args"):
            parsed = _maybe_json_load(first.args)
            if parsed:
                logger.debug("Extracted from tool_calls.args: %s", str(parsed))
                return parsed
        if isinstance(first, dict):
            for key in ["args", "arguments"]:
                if key in first:
                    parsed = _maybe_json_load(first[key])
                    if parsed:
                        logger.debug("Extracted from tool_calls[%s]: %s", key, str(parsed))
                        return parsed
            if "function" in first and isinstance(first["function"], dict):
                parsed = _maybe_json_load(first["function"].get("arguments"))
                if parsed:
                    logger.debug("Extracted from tool_calls.function.arguments: %s", str(parsed))
                    return parsed

    # Try content field (may contain JSON string)
    content = getattr(raw, "content", None)
    if content:
        if isinstance(content, str):
            logger.debug("Attempting to extract JSON from string content: %s", content)
            parsed = _maybe_json_load(content)
            if parsed:
                logger.debug("Successfully parsed JSON from content string")
                return parsed
        elif isinstance(content, dict):
            logger.debug("Content is already a dict")
            return content

    # Check additional_kwargs for any structured data
    if hasattr(raw, "additional_kwargs"):
        kwargs = raw.additional_kwargs
        if isinstance(kwargs, dict):
            if "parsed" in kwargs:
                parsed_data = kwargs["parsed"]
                if isinstance(parsed_data, dict):
                    logger.debug("Found parsed data in additional_kwargs")
                    return parsed_data

    logger.debug(
        "Raw extraction failed. Raw type: %s, has content: %s, has tool_calls: %s",
        type(raw).__name__, bool(content), bool(tool_calls),
    )
    return None


def _unwrap_envelope(parsed: Any) -> Dict[str, Any]:
    """Unwrap message from envelope if present. Always returns a dict."""
    if parsed is None:
        return {}

    # Handle Pydantic models with message attribute (RequestMsgEnvelope)
    if hasattr(parsed, "message"):
        msg = parsed.message
        if hasattr(msg, "model_dump"):
            return msg.model_dump()
        if isinstance(msg, dict):
            return msg
        if isinstance(msg, str):
            loaded = _maybe_json_load(msg)
            return loaded if loaded else {}
        return {}

    # Handle Pydantic models without message attribute (FinalMsg, etc.)
    if hasattr(parsed, "model_dump"):
        return parsed.model_dump()

    # Handle dictionaries
    if isinstance(parsed, dict):
        if "message" in parsed:
            inner = parsed["message"]
            if isinstance(inner, dict):
                return inner
            if isinstance(inner, str):
                loaded = _maybe_json_load(inner)
                return loaded if loaded else {}
            if hasattr(inner, "model_dump"):
                return inner.model_dump()

        if parsed.get("type") in ("offer", "decline", "request", "final"):
            return parsed

        if "has_exact_match" in parsed or "constraint_quoted" in parsed or "hint" in parsed:
            if "type" not in parsed:
                parsed["type"] = "offer"
            return parsed

        return parsed

    if hasattr(parsed, "__dict__"):
        return vars(parsed)

    return {}


def _is_gemini(llm: BaseChatModel) -> bool:
    """Check if the LLM is a Gemini model."""
    return isinstance(llm, ChatGoogleGenerativeAI)


def _needs_flat_schema(llm: BaseChatModel) -> bool:
    """True for models that need flat schemas (no nested $ref).

    Gemini and open-source models (via OpenRouter or local endpoints)
    often fail with nested Pydantic schemas in function calling.
    Native OpenAI models handle nested schemas fine.
    """
    if _is_gemini(llm):
        return True
    # ChatOpenAI instances with a custom base_url are proxied models
    # (OpenRouter, vLLM, ollama, etc.) -- use flat schemas.
    if isinstance(llm, ChatOpenAI):
        base_url = getattr(llm, "openai_api_base", None) or getattr(llm, "base_url", None)
        if base_url and "api.openai.com" not in str(base_url):
            return True
    return False


# Models (matched as lowercase substrings of the model name) that do not support
# function calling or OpenAI-style structured outputs at all.  These require
# manual JSON extraction via prompt injection.
_NO_FUNCTION_CALLING_PATTERNS = ("qwen", "mistral", "llama", "gemma")


def _needs_manual_json(llm: BaseChatModel) -> bool:
    """True for proxied models that don't support function calling or beta structured outputs."""
    if not isinstance(llm, ChatOpenAI):
        return False
    base_url = getattr(llm, "openai_api_base", None) or getattr(llm, "base_url", None)
    if not base_url or "api.openai.com" in str(base_url):
        return False
    model_name = (getattr(llm, "model_name", None) or "").lower()
    return any(pat in model_name for pat in _NO_FUNCTION_CALLING_PATTERNS)


def _build_json_template(schema_class: type) -> str:
    """Build a concrete fill-in JSON template from a Pydantic model's JSON schema.

    Shows placeholders like `"<description>"` instead of raw schema metadata so
    the model can't mistake the template for a valid output object.
    """
    schema = schema_class.model_json_schema()
    props = schema.get("properties", {})
    template: dict = {}
    for fname, fprop in props.items():
        if "enum" in fprop:
            template[fname] = fprop["enum"][0]
        elif "const" in fprop:
            template[fname] = fprop["const"]
        elif "default" in fprop:
            template[fname] = fprop["default"]
        else:
            desc = fprop.get("description", fname).split(".")[0][:50].strip()
            template[fname] = f"<{desc}>"
    return json.dumps(template, indent=2)


def _manual_json_call(llm: BaseChatModel, schema_class: type, messages: list) -> dict:
    """Structured output via prompt injection for models that don't support tool/structured APIs.

    Injects a concrete fill-in template (not the raw JSON Schema) into the system
    prompt so the model can't confuse the schema metadata with the expected output.
    """
    template = _build_json_template(schema_class)
    instruction = (
        "\n\nIMPORTANT: Reply with a JSON object that fills in ACTUAL VALUES. "
        "Do NOT return the template itself — replace every <...> placeholder with a real value. "
        f"No explanation, no markdown fences. Use exactly this structure:\n{template}"
    )

    modified: list = []
    injected = False
    for msg in messages:
        if isinstance(msg, dict):
            role, content = msg.get("role"), msg.get("content", "")
        else:
            role = getattr(msg, "type", None)
            content = getattr(msg, "content", "")
        if role == "system" and not injected:
            modified.append({"role": "system", "content": content + instruction})
            injected = True
        else:
            modified.append(msg)
    if not injected:
        modified.insert(0, {"role": "system", "content": instruction.lstrip()})

    raw_response = llm.invoke(modified)
    text = getattr(raw_response, "content", str(raw_response)).strip()

    # Strip optional ```json ... ``` fences
    fence_match = re.match(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence_match:
        text = fence_match.group(1).strip()

    try:
        parsed = schema_class.model_validate(json.loads(text))
        return {"parsed": parsed, "raw": raw_response, "parsing_error": None}
    except Exception as exc:
        return {"parsed": None, "raw": raw_response, "parsing_error": exc}


def _structured_invoke(llm: BaseChatModel, schema_class: type, messages: list) -> dict:
    """Invoke LLM with structured output, using manual JSON for non-function-calling models."""
    if _needs_manual_json(llm):
        return _manual_json_call(llm, schema_class, messages)
    structured_llm = llm.with_structured_output(
        schema_class, method="function_calling", include_raw=True
    )
    return structured_llm.invoke(messages)


class _StructuredInvoker:
    """Wrapper that exposes `.invoke(messages)` using `_structured_invoke` internally.

    Used when `structured_llm` must be created once and called multiple times
    (e.g., the final-answer node which reuses the same object across react /
    reflexion / llm branches).
    """

    def __init__(self, llm: BaseChatModel, schema_class: type) -> None:
        self._llm = llm
        self._schema = schema_class

    def invoke(self, messages: list) -> dict:
        return _structured_invoke(self._llm, self._schema, messages)


def _request_schema(llm: BaseChatModel):
    """Return the appropriate request schema for the LLM provider."""
    return RequestMsgGemini if _needs_flat_schema(llm) else RequestMsgEnvelope


def _final_schema(llm: BaseChatModel):
    """Return the appropriate final answer schema for the LLM provider."""
    return FinalMsgGemini if _needs_flat_schema(llm) else FinalMsg


def _make_llm(
        model: str,
        temperature: float = 0.0,
        seed: Optional[int] = None,
        timeout: int = 120,
) -> BaseChatModel:
    """Create a chat model instance.

    Model name prefixes determine the backend:
      - ``gemini-*``         -> Google GenAI (ChatGoogleGenerativeAI)
      - ``openrouter/<model>`` -> OpenRouter (ChatOpenAI with base_url)
      - ``local/<model>``    -> Local OpenAI-compatible server (ChatOpenAI with base_url)
      - anything else        -> OpenAI API (ChatOpenAI)

    Environment variables:
      - OPENROUTER_API_KEY   : API key for OpenRouter
      - MIRA_MATH_LOCAL_BASE_URL : Base URL for local server (default: http://localhost:8000/v1)
    """
    if model.startswith("gemini"):
        return ChatGoogleGenerativeAI(
            model=model,
            temperature=temperature,
            timeout=timeout,
            max_retries=1,
        )

    if model.startswith("openrouter/"):
        actual_model = model[len("openrouter/"):]
        api_key = os.environ.get("OPENROUTER_API_KEY", "")
        return ChatOpenAI(
            model=actual_model,
            temperature=temperature,
            seed=seed,
            timeout=timeout,
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )

    if model.startswith("local/"):
        actual_model = model[len("local/"):]
        base_url = os.environ.get("MIRA_MATH_LOCAL_BASE_URL", "http://localhost:8000/v1")
        api_key = os.environ.get("MIRA_MATH_LOCAL_API_KEY", "not-needed")
        return ChatOpenAI(
            model=actual_model,
            temperature=temperature,
            seed=seed,
            timeout=float(os.environ.get("MIRA_MATH_LOCAL_TIMEOUT", 120)),
            base_url=base_url,
            api_key=api_key,
        )

    try:
        return ChatOpenAI(model=model, temperature=temperature, seed=seed, timeout=timeout)
    except TypeError:
        model_kwargs = {"seed": seed} if seed else None
        return ChatOpenAI(model=model, temperature=temperature, model_kwargs=model_kwargs, timeout=timeout)


def _get_tools(tool_names: Optional[List[str]] = None) -> List:
    """Get tool instances by name."""
    available_tools = {
        "calculator": CALCULATOR_TOOL,
    }
    if tool_names is None:
        return []
    return [available_tools[name] for name in tool_names if name in available_tools]


def _build_retry_guidance(declined_requests: List[str]) -> str:
    """Build retry guidance for Agent A listing what B has already declined.

    Agent B does not provide hints about what it has — it only confirms or
    declines. So this guidance solely lists what has been tried and failed,
    so A does not waste attempts repeating them.
    """
    if not declined_requests:
        return ""

    declined_lines = "\n".join(f'  - "{r}"' for r in declined_requests)

    return (
        f"\n\n## ⚠ NEGOTIATION HISTORY — READ CAREFULLY BEFORE REQUESTING\n"
        f"Agent B has DECLINED the following requests (do NOT ask for these again):\n"
        f"{declined_lines}\n"
        f"\n→ You MUST ask for a DIFFERENT type of information this time.\n"
        f"  Think about what other constraints Agent B might have that are different\n"
        f"  from what you already have and different from what you already tried."
    )


# ---------------------------------------------------------------------------
# Verbose Output Helper
# ---------------------------------------------------------------------------
def _verbose_print(verbose: bool, header: str, content: str) -> None:
    """Print verbose output with full content — no truncation."""
    if not verbose:
        return
    print(f"\n{'─' * 60}")
    print(f"  [VERBOSE] {header}")
    print(f"{'─' * 60}")
    for line in content.split('\n'):
        print(f"  {line}")
    print(f"{'─' * 60}\n")


# ---------------------------------------------------------------------------
# LLM Method (Single-shot)
# ---------------------------------------------------------------------------
def _agent_a_request_llm(
        state: MASState,
        llm: BaseChatModel,
        tools: List,
        logger: logging.Logger,
) -> Dict[str, Any]:
    """Agent A makes a request using simple LLM invocation."""
    instance = state["instance"]
    shared_history = state["shared_history"]
    request_count = state["request_count"] + 1
    verbose = state.get("verbose", False)
    declined_requests = state.get("declined_requests", [])

    sys_msg = system_prompt_agent_a()
    user_msg = user_prompt_agent_a(
        instance,
        shared_history,
        attempt_num=request_count,
        max_attempts=state["max_requests"],
    )

    user_msg += _build_retry_guidance(declined_requests)

    if tools:
        tool_names = [t.name for t in tools]
        user_msg += f"\n\nYou have access to these tools: {tool_names}. Use them if helpful for calculations."

    messages = [
        {"role": "system", "content": sys_msg},
        {"role": "user", "content": user_msg},
    ]

    logger.info("Agent A (LLM) request attempt %d/%d", request_count, state["max_requests"])

    if verbose:
        _verbose_print(verbose, "LLM METHOD - Request Phase",
                       f"Method: SINGLE-SHOT LLM (no reasoning loop)\n"
                       f"Attempt: {request_count}/{state['max_requests']}\n"
                       f"Declined so far: {declined_requests}\n"
                       f"Process: Single structured output call to LLM\n"
                       f"No internal reasoning loop - direct prompt → response")

    result = _structured_invoke(llm, _request_schema(llm), messages)

    parsed = result.get("parsed")
    raw = result.get("raw")
    parsing_error = result.get("parsing_error")

    if parsed is None and raw is not None:
        logger.warning(
            "Agent A request: structured parse failed (%s), attempting raw extraction",
            str(parsing_error) if parsing_error else "no parsing_error",
        )
        extracted = _extract_from_raw(raw, logger)
        parsed_dict = _unwrap_envelope(extracted) if extracted else {"type": "error"}
    else:
        parsed_dict = _unwrap_envelope(parsed)

    if verbose:
        _verbose_print(verbose, "LLM METHOD - Response",
                       f"Reasoning: {parsed_dict.get('reasoning', 'N/A')}\n"
                       f"Request: {parsed_dict.get('request', 'N/A')}")

    if parsed_dict.get("type") != "request":
        logger.warning("Agent A request: expected type='request', got '%s'", parsed_dict.get("type"))
        parsed_dict = {
            "type": "request",
            "reasoning": "Unable to parse proper request",
            "request": "unknown",
            "progress": {},
        }

    if not parsed_dict.get("progress"):
        parsed_dict["progress"] = {}

    return {
        "parsed_dict": parsed_dict,
        "parsing_error": parsing_error,
        "request_count": request_count,
    }


# ---------------------------------------------------------------------------
# ReAct Method (Reasoning + Acting loop)
# ---------------------------------------------------------------------------
def _agent_a_request_react(
        state: MASState,
        llm: BaseChatModel,
        tools: List,
        logger: logging.Logger,
) -> Dict[str, Any]:
    """Agent A makes a request using ReAct agent with tools.

    Two-phase approach:
    1. ReAct reasoning phase: Agent reasons freely with tools (no truncation)
    2. Structured output phase: Convert reasoning into proper request format
    """
    instance = state["instance"]
    shared_history = state["shared_history"]
    request_count = state["request_count"] + 1
    verbose = state.get("verbose", False)
    declined_requests = state.get("declined_requests", [])

    sys_msg = system_prompt_agent_a()
    user_msg = user_prompt_agent_a(
        instance,
        shared_history,
        attempt_num=request_count,
        max_attempts=state["max_requests"],
    )

    user_msg += _build_retry_guidance(declined_requests)

    mode_label = "ReAct" if tools else "Two-pass CoT"
    logger.info("Agent A (%s) request attempt %d/%d", mode_label, request_count, state["max_requests"])

    if verbose:
        if tools:
            method_desc = (
                f"Method: ReAct (Reasoning + Acting)\n"
                f"Tools available: {[t.name for t in tools]}\n"
                f"Process:\n"
                f"  Phase 1: THOUGHT → ACTION → OBSERVATION loop (with tools)\n"
                f"  Phase 2: Convert full reasoning trace to structured output"
            )
        else:
            method_desc = (
                f"Method: Two-pass CoT (react mode, no tools)\n"
                f"Tools available: None\n"
                f"Process:\n"
                f"  Phase 1: Single chain-of-thought reasoning call\n"
                f"  Phase 2: Convert reasoning into structured output"
            )
        _verbose_print(verbose, "ReAct METHOD - Request Phase",
                       f"{method_desc}\n"
                       f"Attempt: {request_count}/{state['max_requests']}\n"
                       f"Declined so far: {declined_requests}")

    react_reasoning_prompt = """
## ReAct Reasoning Phase
Think step-by-step to analyze this problem:

1. THOUGHT: What constraints do I have? What can I determine?
2. ACTION: Use calculator tool if you need to compute anything
3. OBSERVATION: Note the result
4. Repeat THOUGHT/ACTION/OBSERVATION as needed
5. CONCLUSION: What specific information do I need from the other agent?

Reason thoroughly. Use tools as needed. Take your time to analyze fully.
"""

    full_reasoning_trace = ""
    reasoning_parts = []

    if tools:
        react_agent = create_react_agent(llm, tools)

        if verbose:
            print(f"  [ReAct] Starting THOUGHT/ACTION/OBSERVATION loop with tools...")

        result = react_agent.invoke(
            {
                "messages": [
                    SystemMessage(content=sys_msg + react_reasoning_prompt),
                    HumanMessage(content=user_msg),
                ]
            },
            config={"recursion_limit": 10},
        )

        messages = result.get("messages", [])
        for msg in messages:
            if hasattr(msg, "content") and msg.content:
                msg_type = type(msg).__name__
                if msg_type == "AIMessage":
                    reasoning_parts.append(f"THOUGHT: {msg.content}")
                elif msg_type == "ToolMessage":
                    reasoning_parts.append(f"OBSERVATION: {msg.content}")

        full_reasoning_trace = "\n".join(reasoning_parts)
        logger.info("ReAct completed %d reasoning steps", len(reasoning_parts))
    else:
        if verbose:
            print(f"  [ReAct] No tools - using chain-of-thought reasoning...")

        cot_messages = [
            {"role": "system", "content": sys_msg + react_reasoning_prompt},
            {"role": "user", "content": user_msg},
        ]

        response = llm.invoke(cot_messages)
        full_reasoning_trace = response.content if hasattr(response, "content") else str(response)
        reasoning_parts = [full_reasoning_trace]
        logger.info("Chain-of-thought reasoning completed")

    if verbose:
        _verbose_print(verbose, f"ReAct Phase 1 - Reasoning Trace ({len(reasoning_parts)} steps)",
                       full_reasoning_trace)

    structured_prompt = f"""Based on your reasoning below, provide a structured request.

## Your Reasoning
{full_reasoning_trace}

## Task
Now output ONLY a JSON object with your request:
{{
    "type": "request",
    "reasoning": "<summarize your key findings>",
    "request": "<specific info you need from other agent>",
    "progress": {{"expressions": {{...}} or "candidates": [...]}}
}}

Be specific in your request. What exactly do you need?
"""

    result = _structured_invoke(llm, _request_schema(llm), [
        {"role": "system", "content": "Convert the reasoning into a structured request."},
        {"role": "user", "content": structured_prompt},
    ])

    parsed = result.get("parsed")
    raw = result.get("raw")

    if parsed is None and raw is not None:
        extracted = _extract_from_raw(raw, logger)
        parsed_dict = _unwrap_envelope(extracted) if extracted else {}
    else:
        parsed_dict = _unwrap_envelope(parsed)

    if parsed_dict.get("type") != "request":
        parsed_dict = {
            "type": "request",
            "reasoning": full_reasoning_trace,
            "request": "unknown",
            "progress": {},
        }

    if "reasoning" not in parsed_dict or not parsed_dict["reasoning"]:
        parsed_dict["reasoning"] = full_reasoning_trace

    if verbose:
        _verbose_print(verbose, "ReAct Phase 2 - Structured Output",
                       f"Request: {parsed_dict.get('request', 'N/A')}\n"
                       f"Reasoning summary: {parsed_dict.get('reasoning', 'N/A')}")

    if not parsed_dict.get("progress"):
        parsed_dict["progress"] = {}

    return {
        "parsed_dict": parsed_dict,
        "parsing_error": None,
        "request_count": request_count,
        "full_reasoning_trace": full_reasoning_trace,
    }


# ---------------------------------------------------------------------------
# Reflexion Method (Self-reflection loop)
# ---------------------------------------------------------------------------
def _agent_a_request_reflexion(
        state: MASState,
        llm: BaseChatModel,
        tools: List,
        logger: logging.Logger,
) -> Dict[str, Any]:
    """Agent A makes a request using Reflexion (self-reflection loop)."""
    instance = state["instance"]
    shared_history = state["shared_history"]
    request_count = state["request_count"] + 1
    verbose = state.get("verbose", False)
    declined_requests = state.get("declined_requests", [])

    sys_msg = system_prompt_agent_a()
    user_msg = user_prompt_agent_a(
        instance,
        shared_history,
        attempt_num=request_count,
        max_attempts=state["max_requests"],
    )

    user_msg += _build_retry_guidance(declined_requests)

    logger.info("Agent A (Reflexion) request attempt %d/%d", request_count, state["max_requests"])

    max_reflections = 2

    if verbose:
        _verbose_print(verbose, "REFLEXION METHOD - Request Phase",
                       f"Method: Reflexion (Generate → Reflect → Refine)\n"
                       f"Attempt: {request_count}/{state['max_requests']}\n"
                       f"Reflection iterations: {max_reflections}\n"
                       f"Declined so far: {declined_requests}\n"
                       f"Process:\n"
                       f"  Step 1: Generate initial response\n"
                       f"  Step 2: Self-reflect and critique\n"
                       f"  Step 3: Refine based on reflection\n"
                       f"  (Repeat steps 2-3 for {max_reflections} iterations)")

    initial_prompt = user_msg + "\n\nGenerate your initial analysis and request. Be thorough."

    messages = [
        {"role": "system", "content": sys_msg},
        {"role": "user", "content": initial_prompt},
    ]

    result = _structured_invoke(llm, _request_schema(llm), messages)

    parsed = result.get("parsed")
    raw = result.get("raw")

    if parsed is None and raw is not None:
        extracted = _extract_from_raw(raw, logger)
        current_response = _unwrap_envelope(extracted) if extracted else {}
    else:
        current_response = _unwrap_envelope(parsed)

    if verbose:
        _verbose_print(verbose, "Reflexion Step 1 - Initial Generation",
                       f"Request: {current_response.get('request', 'N/A')}\n"
                       f"Reasoning: {current_response.get('reasoning', 'N/A')}")

    for i in range(max_reflections):
        logger.info("Agent A (Reflexion) reflection %d/%d", i + 1, max_reflections)

        if verbose:
            print(f"  [Reflexion] Starting reflection iteration {i + 1}/{max_reflections}...")

        reflection_prompt = f"""
Your previous response was:
{json.dumps(current_response, indent=2)}

## Self-Reflection
1. Is your reasoning complete and correct?
2. Did you identify all constraints and unknowns?
3. Is your request specific enough?
4. Could the other agent have exactly what you're asking for?
5. Have you already tried this request and been declined? If so, ask for something different.

If you find issues, provide an improved response. If your response is good, repeat it.
"""

        reflection_messages = [
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": json.dumps(current_response)},
            {"role": "user", "content": reflection_prompt},
        ]

        result = structured_llm.invoke(reflection_messages)
        parsed = result.get("parsed")
        raw = result.get("raw")

        if parsed is None and raw is not None:
            extracted = _extract_from_raw(raw, logger)
            refined = _unwrap_envelope(extracted) if extracted else current_response
        else:
            refined = _unwrap_envelope(parsed)

        prev_request = current_response.get("request", "")
        if refined and refined.get("type") == "request":
            current_response = refined

        if verbose:
            new_request = current_response.get("request", "")
            changed = "CHANGED" if new_request != prev_request else "UNCHANGED"
            _verbose_print(verbose, f"Reflexion Iteration {i + 1} - {changed}",
                           f"Previous request: {prev_request}\n"
                           f"Current request: {new_request}\n"
                           f"Reasoning: {current_response.get('reasoning', 'N/A')}")

    parsed_dict = current_response

    if parsed_dict.get("type") != "request":
        parsed_dict = {
            "type": "request",
            "reasoning": "Reflexion analysis",
            "request": "unknown",
            "progress": {},
        }

    if not parsed_dict.get("progress"):
        parsed_dict["progress"] = {}

    if verbose:
        _verbose_print(verbose, "Reflexion - Final Output",
                       f"Request: {parsed_dict.get('request', 'N/A')}\n"
                       f"Total reflection iterations: {max_reflections}")

    return {
        "parsed_dict": parsed_dict,
        "parsing_error": None,
        "request_count": request_count,
    }


# ---------------------------------------------------------------------------
# Agent A Final (same for all methods)
# ---------------------------------------------------------------------------
def _agent_a_final_impl(
        state: MASState,
        llm: BaseChatModel,
        tools: List,
        logger: logging.Logger,
        method: str,
) -> Dict[str, Any]:
    """Agent A computes and submits final answer.

    KEY FIX: Uses FinalMsg directly with with_structured_output instead of
    FinalMsgEnvelope. FinalMsg is already flat (type, reasoning, answer,
    consistency_check) so the LLM fills top-level fields directly. The old
    FinalMsgEnvelope wrapped FinalMsg in `message: FinalMsg`, causing the LLM to
    put a text string in `message` instead of a nested dict — same bug that was
    fixed for Agent B. With FinalMsg used directly, structured parsing succeeds
    reliably.
    """
    instance = state["instance"]
    shared_history = state["shared_history"]
    verbose = state.get("verbose", False)

    sys_msg = system_prompt_agent_a()
    user_msg = user_prompt_agent_a(
        instance,
        shared_history,
        attempt_num=state["request_count"],
        max_attempts=state["max_requests"],
    )
    user_msg += (
        "\n\n## FINAL ANSWER PHASE — Solve the System\n\n"
        "You now have ALL the information. Do the following steps IN ORDER. "
        "Do not skip any step.\n\n"

        "**STEP A — List every known value.**\n"
        "Write a table: for every variable in your constraints that has a direct "
        "assignment (including the value just received from Agent B), write:\n"
        "  variable = number\n"
        "Do this for ALL of them before touching any equation.\n\n"

        "**STEP B — Substitute into every system equation.**\n"
        "Take each system equation (e.g. '4 u_1_1 = u_0_1 + u_2_1 + u_1_0 + u_1_2').\n"
        "Replace EVERY variable that has a known value with its number from Step A.\n"
        "Rearrange to the form:  coeff*X + coeff*Y + ... = number  (all unknowns on left, numbers on right).\n"
        "Write each equation out completely in this standard form.\n\n"

        "**STEP C — Write the augmented matrix.**\n"
        "Order your remaining unknowns (e.g. u_1_1, u_1_2, u_2_1, u_2_2).\n"
        "Write one row per equation, coefficients in that order, then | RHS:\n"
        "  [ c1  c2  c3  c4 | rhs ]\n"
        "  [ c1  c2  c3  c4 | rhs ]\n"
        "  ...\n"
        "If you have more unknowns than rows, STOP — you are still missing information.\n\n"

        "**STEP D — Gaussian elimination.**\n"
        "Use row operations on the augmented matrix to reach row-echelon form.\n"
        "Show each row operation explicitly: e.g. R2 = R2 + (1/4)*R1.\n"
        "Back-substitute to find each unknown.\n\n"

        "**STEP E — Verify.**\n"
        "Plug every solved value back into EVERY original equation from Step B.\n"
        "Check that both sides are equal. If any equation fails, your arithmetic "
        "is wrong — find the error and redo.\n\n"

        "**STEP F — Report.**\n"
        "State the value of the TARGET variable from Step 2 of your prompt.\n"
        "Report ONLY that value as the final answer JSON. "
        "Do NOT report the hint B gave you or any intermediate variable."
    )

    if tools:
        tool_names = [t.name for t in tools]
        user_msg += f"\n\nUse these tools if helpful: {tool_names}"

    logger.info("Agent A (%s) computing final answer", method.upper())

    if verbose:
        if method == "react" and not tools:
            final_method_desc = "Two-pass CoT (react mode, no tools)"
        else:
            final_method_desc = method.upper()
        _verbose_print(verbose, f"{method.upper()} METHOD - Final Answer Phase",
                       f"Method: {final_method_desc}\n"
                       f"Tools available: {[t.name for t in tools] if tools else 'None'}\n"
                       f"Computing final answer...")

    # Use FinalMsg directly — no envelope wrapper
    structured_llm = _StructuredInvoker(llm, _final_schema(llm))

    if method == "react":
        # Phase 1: Full reasoning (with tools if available)
        reasoning_instructions = """
## IMPORTANT: This is the FINAL ANSWER phase. You CANNOT request more information from Agent B.

You already received information from Agent B during the exchange above.
Use ALL information you have — your private constraints AND everything Agent B provided — to compute the answer NOW.

Steps:
1. Identify the problem family.
2. Write out all the mathematical expressions/constraints you have (your own + Agent B's hints).
3. Solve the system step-by-step. Show your work.
4. Output ONLY the final answer as the required JSON.

You MUST output a computed answer (a number, dict, etc.), NOT a request for more information.
"""

        if tools:
            react_agent = create_react_agent(llm, tools)
            result = react_agent.invoke(
                {
                    "messages": [
                        SystemMessage(content=sys_msg + reasoning_instructions),
                        HumanMessage(content=user_msg),
                    ]
                },
                config={"recursion_limit": 10},
            )
            messages = result.get("messages", [])
            reasoning_parts = []
            for msg in messages:
                if hasattr(msg, "content") and msg.content:
                    msg_type = type(msg).__name__
                    if msg_type == "AIMessage":
                        reasoning_parts.append(f"THOUGHT: {msg.content}")
                    elif msg_type == "ToolMessage":
                        reasoning_parts.append(f"OBSERVATION: {msg.content}")
            full_reasoning_trace = "\n".join(reasoning_parts)
        else:
            cot_messages = [
                {"role": "system", "content": sys_msg + reasoning_instructions},
                {"role": "user", "content": user_msg},
            ]
            response = llm.invoke(cot_messages)
            full_reasoning_trace = response.content if hasattr(response, "content") else str(response)

        logger.info("ReAct final answer: completed reasoning phase")

        if verbose:
            _verbose_print(verbose, "ReAct Final - Phase 1 Reasoning Trace", full_reasoning_trace)

        # Phase 2: Convert reasoning to structured FinalMsg
        structured_prompt = f"""Based on your reasoning below, provide the final answer.

## Your Reasoning and Computations
{full_reasoning_trace}

## Task
Output the final answer using the FinalMsg function. The answer field must be a
COMPUTED VALUE (number, dict, list, etc.) — NOT a string asking for more information.
"""

        result = structured_llm.invoke([
            {
                "role": "system",
                "content": (
                    "Extract the computed final answer from the reasoning. "
                    "The answer must be a concrete value, not a request for more information."
                ),
            },
            {"role": "user", "content": structured_prompt},
        ])

        parsed = result.get("parsed")
        raw = result.get("raw")

        if parsed is None and raw is not None:
            logger.warning(
                "ReAct final: structured parse failed (%s), attempting raw extraction",
                str(result.get("parsing_error")) if result.get("parsing_error") else "unknown",
            )
            extracted = _extract_from_raw(raw, logger)
            parsed_dict = _unwrap_envelope(extracted) if extracted else {}
        else:
            parsed_dict = _unwrap_envelope(parsed)

        if not parsed_dict or not parsed_dict.get("answer"):
            parsed_dict = {"type": "final", "answer": None, "reasoning": full_reasoning_trace}

        if verbose:
            _verbose_print(verbose, "ReAct Final - Phase 2 Output",
                           f"Answer: {parsed_dict.get('answer', 'N/A')}\n"
                           f"Reasoning: {parsed_dict.get('reasoning', 'N/A')}")

    elif method == "reflexion":
        logger.info("Reflexion final answer: starting generation-reflection loop")
        max_reflections = 2

        if verbose:
            print(f"  [Reflexion Final] Starting {max_reflections} reflection iterations...")

        initial_prompt = user_msg + "\n\nGenerate your initial answer. Be thorough."
        messages = [
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": initial_prompt},
        ]

        result = structured_llm.invoke(messages)

        parsed = result.get("parsed")
        raw = result.get("raw")

        if parsed is None and raw is not None:
            logger.warning(
                "Reflexion final initial: structured parse failed (%s), attempting raw extraction",
                str(result.get("parsing_error")) if result.get("parsing_error") else "unknown",
            )
            extracted = _extract_from_raw(raw, logger)
            current_response = _unwrap_envelope(extracted) if extracted else {}
        else:
            current_response = _unwrap_envelope(parsed)

        if verbose:
            _verbose_print(verbose, "Reflexion Final - Initial Answer",
                           f"Answer: {current_response.get('answer', 'N/A')}\n"
                           f"Reasoning: {current_response.get('reasoning', 'N/A')}")

        for i in range(max_reflections):
            logger.info("Reflexion final answer: reflection %d/%d", i + 1, max_reflections)

            if verbose:
                print(f"  [Reflexion Final] Reflection iteration {i + 1}/{max_reflections}...")

            reflection_prompt = f"""
Your previous answer was:
{json.dumps(current_response, indent=2)}

## Self-Reflection
1. Is your reasoning complete and correct?
2. Did you use the information from the other agent correctly?
3. Are your calculations accurate?
4. Does your answer match the required format?

If you find issues, provide an improved response. If your answer is correct, confirm it.
"""

            reflection_messages = [
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": json.dumps(current_response)},
                {"role": "user", "content": reflection_prompt},
            ]

            result = structured_llm.invoke(reflection_messages)
            parsed = result.get("parsed")
            raw = result.get("raw")

            if parsed is None and raw is not None:
                extracted = _extract_from_raw(raw, logger)
                refined = _unwrap_envelope(extracted) if extracted else current_response
            else:
                refined = _unwrap_envelope(parsed)

            prev_answer = current_response.get("answer")
            if refined and refined.get("answer") not in (None, {}, []):
                current_response = refined

            if verbose:
                new_answer = current_response.get("answer")
                changed = "CHANGED" if str(new_answer) != str(prev_answer) else "UNCHANGED"
                _verbose_print(verbose, f"Reflexion Final - Iteration {i + 1} - {changed}",
                               f"Previous answer: {prev_answer}\n"
                               f"Current answer: {new_answer}")

        parsed_dict = current_response
        if not parsed_dict.get("type"):
            parsed_dict["type"] = "final"

        if verbose:
            _verbose_print(verbose, "Reflexion Final - Output",
                           f"Final answer: {parsed_dict.get('answer', 'N/A')}\n"
                           f"Total iterations: {max_reflections}")

    else:
        # Standard LLM: single structured FinalMsg call (no envelope)
        if verbose:
            print(f"  [LLM Final] Single structured output call using FinalMsg directly (no envelope)...")

        messages = [
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": user_msg},
        ]

        result = structured_llm.invoke(messages)

        parsed = result.get("parsed")
        raw = result.get("raw")
        parsing_error = result.get("parsing_error")

        if parsed is None and raw is not None:
            logger.warning(
                "LLM final: structured parse failed (%s), attempting raw extraction",
                str(parsing_error) if parsing_error else "unknown",
            )
            extracted = _extract_from_raw(raw, logger)
            parsed_dict = _unwrap_envelope(extracted) if extracted else {}
        else:
            parsed_dict = _unwrap_envelope(parsed)

        if verbose:
            _verbose_print(verbose, "LLM Final - Output",
                           f"Answer: {parsed_dict.get('answer', 'N/A')}\n"
                           f"Reasoning: {parsed_dict.get('reasoning', 'N/A')}")

    # Ensure required fields are present regardless of path taken
    parsed_dict["type"] = "final"
    answer = parsed_dict.get("answer")
    answer_missing = answer is None or answer == {} or answer == []
    if answer_missing:
        # Try alternate keys first
        for key in ["result", "solution", "value", "final_answer"]:
            if key in parsed_dict and parsed_dict[key] is not None:
                parsed_dict["answer"] = parsed_dict[key]
                answer_missing = False
                break
        # Try to extract a JSON object/value from the reasoning text
        if answer_missing:
            parsed_dict["answer"] = _extract_answer_from_reasoning(
                parsed_dict.get("reasoning", ""), logger
            )
    if "reasoning" not in parsed_dict:
        parsed_dict["reasoning"] = ""

    return parsed_dict


# ---------------------------------------------------------------------------
# Agent Node Functions
# ---------------------------------------------------------------------------
def agent_a_request(
        state: MASState,
        llm: BaseChatModel,
        tools: List,
        logger: logging.Logger,
) -> MASState:
    """Agent A makes a request for information."""
    method = state.get("method", "llm")
    shared_history = list(state["shared_history"])
    transcript = list(state["transcript"])

    if method == "react":
        result = _agent_a_request_react(state, llm, tools, logger)
    elif method == "reflexion":
        result = _agent_a_request_reflexion(state, llm, tools, logger)
    else:
        result = _agent_a_request_llm(state, llm, tools, logger)

    parsed_dict = result["parsed_dict"]
    request_count = result["request_count"]
    parsing_error = result.get("parsing_error")

    logger.info("Agent A request: %s", parsed_dict.get("request", ""))

    transcript.append({
        "agent_id": "A",
        "round": request_count - 1,
        "type": "request",
        "message": json.dumps(parsed_dict),
        "parsed": parsed_dict,
        "parsing_error": str(parsing_error) if parsing_error else None,
        "method": method,
    })
    shared_history.append({"from": "A", "message": parsed_dict})

    return {
        **state,
        "shared_history": shared_history,
        "transcript": transcript,
        "request_count": request_count,
        "last_request": parsed_dict,
        "phase": "respond",
    }


def agent_b_respond(
        state: MASState,
        llm: BaseChatModel,
        logger: logging.Logger,
) -> MASState:
    """Agent B responds to A's request (offer or decline).

    Uses AgentBFlatResponse — a flat discriminated schema — instead of the old
    AgentBMsgEnvelope whose `message: Union[OfferMsg, DeclineMsg]` field caused
    the LLM to fill `message` with a plain string, breaking Pydantic validation
    on every call and making B always appear to decline.

    Agent B has two valid responses only:
      - offer:   has the exact information → provide it
      - decline: does not have it → say so, nothing more
    """
    instance = state["instance"]
    shared_history = list(state["shared_history"])
    transcript = list(state["transcript"])
    last_request = state["last_request"]
    declined_requests = list(state.get("declined_requests", []))

    sys_msg = system_prompt_agent_b()
    user_msg = user_prompt_agent_b(instance, last_request or {})

    json_instructions = """

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CRITICAL OUTPUT FORMAT REQUIREMENTS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

You MUST call the provided function (AgentBFlatResponse) with the correct fields.

IF you have an EXACT match for the request, use type="offer":
  type              → "offer"
  has_exact_match   → true
  constraint_quoted → copy the EXACT constraint text from your private data
  hint              → the specific value, e.g. "a(1) = 7" or "sign(x0) = +1"
  decline_message   → null

IF you do NOT have the requested information, use type="decline":
  type              → "decline"
  has_exact_match   → false
  constraint_quoted → null
  hint              → null
  decline_message   → "I do not have that specific information."

RULES:
- ONLY set type="offer" when you can quote an EXACT constraint from your private data.
- Do NOT invent or guess information.
- Do NOT set has_exact_match=true if you are guessing or approximating.
- Do NOT provide any hints or clues about what you do have when declining.
  Simply state that you do not have the requested information.
"""

    user_msg += json_instructions

    messages = [
        {"role": "system", "content": sys_msg},
        {"role": "user", "content": user_msg},
    ]

    logger.info("Agent B responding to request: %s", (last_request or {}).get("request", ""))

    result = _structured_invoke(llm, AgentBFlatResponse, messages)

    parsed: Optional[AgentBFlatResponse] = result.get("parsed")
    raw = result.get("raw")
    parsing_error = result.get("parsing_error")

    if parsed is not None:
        parsed_dict = parsed.to_dict()
        logger.debug("Agent B structured output parsed successfully: type=%s", parsed.type)
    else:
        logger.warning(
            "Agent B structured parse failed (%s), attempting raw extraction",
            str(parsing_error) if parsing_error else "unknown error",
        )

        extracted = _extract_from_raw(raw, logger)

        if extracted:
            try:
                flat = AgentBFlatResponse(**extracted)
                parsed_dict = flat.to_dict()
                logger.info("Agent B: raw extraction + coercion succeeded: type=%s", flat.type)
            except Exception as coerce_err:
                logger.warning("Agent B: coercion failed (%s), building decline from raw", coerce_err)
                parsed_dict = _build_decline_from_raw(raw, logger)
        else:
            parsed_dict = _build_decline_from_raw(raw, logger)

    msg_type = parsed_dict.get("type", "decline")

    # Safety: if offer but has_exact_match is False, treat as decline
    if msg_type == "offer" and not parsed_dict.get("has_exact_match", False):
        logger.warning("Agent B said offer but has_exact_match=False, treating as decline")
        msg_type = "decline"
        parsed_dict = {
            "type": "decline",
            "message": "I do not have that exact information.",
        }

    # Ensure decline has a non-empty message
    if msg_type == "decline" and not parsed_dict.get("message"):
        parsed_dict["message"] = "I do not have that specific information."

    got_hint = msg_type == "offer"

    if got_hint:
        logger.info(
            "Agent B offered: hint=%s | quoted=%s",
            parsed_dict.get("hint", "?"),
            parsed_dict.get("constraint_quoted", "?"),
        )
    else:
        logger.info("Agent B declined: %s", parsed_dict.get("message", ""))
        req_text = (last_request or {}).get("request", "")
        if req_text and req_text not in declined_requests:
            declined_requests.append(req_text)

    transcript.append({
        "agent_id": "B",
        "round": state["request_count"] - 1,
        "type": msg_type,
        "message": json.dumps(parsed_dict),
        "parsed": parsed_dict,
        "parsing_error": str(parsing_error) if parsing_error else None,
    })
    shared_history.append({"from": "B", "message": parsed_dict})

    if got_hint:
        next_phase = "final"
    elif state["request_count"] >= state["max_requests"]:
        next_phase = "failed"
        logger.warning("Max requests reached without getting hint")
    else:
        next_phase = "request"

    return {
        **state,
        "shared_history": shared_history,
        "transcript": transcript,
        "got_hint": got_hint,
        "phase": next_phase,
        "declined_requests": declined_requests,
    }


def _build_decline_from_raw(raw: Any, logger: logging.Logger) -> Dict[str, Any]:
    """Last-resort: build a structured decline dict from whatever the LLM returned."""
    content = getattr(raw, "content", None)
    if content and isinstance(content, str):
        lower = content.lower()
        if any(p in lower for p in ["do not have", "don't have", "cannot provide",
                                     "not available", "no information", "decline"]):
            logger.info("Agent B: plain-text decline detected, converting to structured format")
            return {
                "type": "decline",
                "message": content.strip(),
            }
        parsed_json = _maybe_json_load(content)
        if parsed_json:
            logger.info("Agent B: extracted JSON from plain string content")
            try:
                flat = AgentBFlatResponse(**parsed_json)
                return flat.to_dict()
            except Exception:
                pass

    logger.warning("Agent B: could not extract valid response, defaulting to decline")
    return {
        "type": "decline",
        "message": "I do not have that specific information.",
    }


def _answer_looks_like_request(answer: Any) -> bool:
    """Detect if a 'final answer' is actually a request for more information."""
    if not isinstance(answer, str):
        return False
    lower = answer.lower().strip()
    request_patterns = [
        "the sign of", "the value of", "the measurement",
        "the boundary value", "the matrix entry", "the cell value",
        "the point p(", "the congruence", "the k-th moment",
        "the line equation", "the coordinates", "the path sum",
        "the edge weight", "an equation involving",
        "i need", "i request", "please provide", "i would like to know",
        "is needed", "still need", "request:",
    ]
    return any(p in lower for p in request_patterns)


def agent_a_final(
        state: MASState,
        llm: BaseChatModel,
        tools: List,
        logger: logging.Logger,
) -> MASState:
    """Agent A computes and submits final answer."""
    method = state.get("method", "llm")
    transcript = list(state["transcript"])

    parsed_dict = _agent_a_final_impl(state, llm, tools, logger, method)

    answer = parsed_dict.get("answer")
    if _answer_looks_like_request(answer):
        logger.warning(
            "Final answer looks like a request, not a computed value: %s",
            str(answer),
        )
        parsed_dict["answer_warning"] = "Model output a request instead of a computed answer"

    logger.info("Agent A final answer: %s", str(parsed_dict.get("answer", "")))

    transcript.append({
        "agent_id": "A",
        "round": state["request_count"],
        "type": "final",
        "message": json.dumps(parsed_dict),
        "parsed": parsed_dict,
        "method": method,
    })

    return {
        **state,
        "transcript": transcript,
        "phase": "done",
    }


def agent_a_failed(
        state: MASState,
        logger: logging.Logger,
) -> MASState:
    """Handle case where A exhausted all attempts without getting hint."""
    transcript = list(state["transcript"])
    logger.warning(
        "Agent A failed to get required information within %d attempts",
        state["max_requests"],
    )
    transcript.append({
        "agent_id": "A",
        "round": state["request_count"],
        "type": "failed",
        "message": json.dumps({"type": "failed", "reason": "max_requests_exceeded"}),
        "parsed": {"type": "failed", "reason": "max_requests_exceeded"},
    })
    return {**state, "transcript": transcript, "phase": "done"}


# ---------------------------------------------------------------------------
# Graph Construction
# ---------------------------------------------------------------------------
def build_graph(
        llm_a: BaseChatModel,
        llm_b: BaseChatModel,
        tools: List,
        logger: logging.Logger,
        verbose: bool = False,
) -> StateGraph:
    """Build the LangGraph for the mira_math protocol."""
    graph = StateGraph(MASState)

    graph.add_node("a_request", lambda s: agent_a_request(s, llm_a, tools, logger))
    graph.add_node("b_respond", lambda s: agent_b_respond(s, llm_b, logger))
    graph.add_node("a_final", lambda s: agent_a_final(s, llm_a, tools, logger))
    graph.add_node("a_failed", lambda s: agent_a_failed(s, logger))

    graph.set_entry_point("a_request")

    def route_after_respond(state: MASState) -> str:
        phase = state["phase"]
        if phase == "final":
            return "a_final"
        elif phase == "failed":
            return "a_failed"
        else:
            return "a_request"

    graph.add_edge("a_request", "b_respond")
    graph.add_conditional_edges("b_respond", route_after_respond)
    graph.add_edge("a_final", END)
    graph.add_edge("a_failed", END)

    return graph


# ---------------------------------------------------------------------------
# Main Runner
# ---------------------------------------------------------------------------

def _run_oracle_instance(
        instance: Dict[str, Any],
        llm_a: BaseChatModel,
        tools: List,
        logger: logging.Logger,
        method: str,
        max_requests: int,
        tool_names: Optional[List[str]],
        verbose: bool,
) -> List[Dict[str, Any]]:
    """Run the oracle-hint condition for one instance.

    Agent A is placed directly in the state a real run reaches after a
    *successful* exchange: a synthetic request/offer pair is written into the
    shared history, carrying the atomic hint in the family's canonical
    notation, and A is asked for its final answer. Agent B is never invoked.

    The hint is delivered through the shared history rather than by rewriting
    A's private constraints. Both routes make the view well-posed, but only
    this one reproduces the prompt framing of a real successful run: with an
    empty history the prompt's "find what to request" scaffolding dominates and
    A emits another request instead of an answer, which measures instruction
    framing rather than solving ability.

    The family's ``apply_hint`` is still used, as a verification gate and to
    derive the canonical hint text, via `build_oracle_instance`.

    Args:
        instance: The mira_math instance to run.
        llm_a: Chat model backing Agent A.
        tools: Tool objects available to Agent A.
        logger: Logger for run diagnostics.
        method: Agent A reasoning method (recorded for transcript parity).
        max_requests: Retry budget, recorded but unreachable in this condition.
        tool_names: Tool names, recorded in the state for parity.
        verbose: Whether to print prompt/response detail.

    Returns:
        A one-entry transcript containing Agent A's final message.
    """
    # Verifies the hint resolves A's ill-posedness and yields its canonical text.
    _verified, injected = build_oracle_instance(instance, agent_id="A", verify=True)
    # Deliver B's own wording: apply_hint's rendering is not always equivalent in
    # difficulty to the constraint B actually holds (see select_b_resolving_constraint).
    hint_texts, hint_source = select_b_resolving_constraint(instance, injected, agent_id="A")
    hint_text = "; ".join(hint_texts)
    request_text = _format_ideal_request(instance)

    logger.info(
        "Oracle condition on %s: replaying offer %r (source=%s) for request %r",
        instance.get("id", "?"), hint_text, hint_source, request_text,
    )

    synthetic_request = {
        "type": "request",
        "reasoning": "Oracle condition: the resolving request is supplied, not inferred.",
        "request": request_text,
    }
    synthetic_offer = {
        "type": "offer",
        "has_exact_match": True,
        "constraint_quoted": hint_text,
        "hint": hint_text,
    }

    state: MASState = {
        "instance": instance,
        "shared_history": [
            {"from": "A", "message": synthetic_request},
            {"from": "B", "message": synthetic_offer},
        ],
        "transcript": [],
        # Must be > 1: prompts.py gates the closing instruction on attempt_num,
        # emitting "identify the missing piece, then request it from Agent B" at
        # attempt 1 and "you now have the missing piece -- solve" afterwards.
        # At 1 the oracle is told to go request what it has already been given.
        "request_count": 2,
        "max_requests": max_requests,
        "last_request": synthetic_request,
        "phase": "final",
        "got_hint": True,
        "method": method,
        "tools_used": tool_names or [],
        "verbose": verbose,
        "declined_requests": [],
    }

    final_state = agent_a_final(state, llm_a, tools, logger)
    transcript = final_state["transcript"]
    for entry in transcript:
        entry["condition"] = CONDITION_NAME
        entry["injected_constraints"] = hint_texts
        entry["apply_hint_rendering"] = injected
        entry["hint_source"] = hint_source
        entry["oracle_request"] = request_text
    return transcript


def run_instance(
        instance: Dict[str, Any],
        model_a: str,
        model_b: str,
        method: str = "llm",
        tool_names: Optional[List[str]] = None,
        temperature: float = 0.0,
        seed: Optional[int] = None,
        logger: Optional[logging.Logger] = None,
        verbose: bool = False,
        oracle: bool = False,
) -> List[Dict[str, Any]]:
    """Run a single mira_math instance and return the transcript.

    Args:
        oracle: If True, run the oracle-hint condition instead of the two-agent
            protocol: Agent A's view is pre-augmented with its atomic hint and
            Agent B is never called. Measures solving ability in isolation.
    """
    if logger is None:
        logger = logging.getLogger("mira_math")

    llm_a = _make_llm(model_a, temperature, seed)
    tools = _get_tools(tool_names)

    difficulty = instance.get("difficulty", 1)
    max_requests = get_max_requests(difficulty)

    if oracle:
        return _run_oracle_instance(
            instance, llm_a, tools, logger, method, max_requests, tool_names, verbose,
        )

    llm_b = _make_llm(model_b, temperature, seed)

    tools_str = ", ".join(tool_names) if tool_names else "None"
    logger.info(
        "Running instance %s (difficulty=%d, max_requests=%d, method=%s, tools=%s)",
        instance.get("id", "?"), difficulty, max_requests, method, tools_str,
    )

    graph = build_graph(llm_a, llm_b, tools, logger, verbose)
    app = graph.compile()

    initial_state: MASState = {
        "instance": instance,
        "shared_history": [],
        "transcript": [],
        "request_count": 0,
        "max_requests": max_requests,
        "last_request": None,
        "phase": "request",
        "got_hint": False,
        "method": method,
        "tools_used": tool_names or [],
        "verbose": verbose,
        "declined_requests": [],
    }

    final_state = app.invoke(initial_state)
    return final_state["transcript"]


def main() -> None:
    parser = argparse.ArgumentParser(description="mira_math Single Instance Runner")
    parser.add_argument("--instance", default="examples/sample_instance.json",
                        help="Path to instance JSON file")
    parser.add_argument("--model-a", default=None, help="Model for Agent A")
    parser.add_argument("--model-b", default=None, help="Model for Agent B")
    parser.add_argument("--method", choices=["llm", "react", "reflexion"], default="llm",
                        help="Agent method: llm (single-shot), react (reasoning+acting), reflexion (self-reflection)")
    parser.add_argument("--tools", nargs="*", default=None,
                        help="Tools to give agents (e.g., calculator)")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--log-dir", default="logs")
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args()

    _load_env()

    model_default = os.environ.get("MIRA_MATH_MODEL", "gpt-4o-mini")
    model_a = args.model_a or os.environ.get("MIRA_MATH_MODEL_TIER_A") or os.environ.get("MIRA_MATH_MODEL_A", model_default)
    model_b = args.model_b or os.environ.get("MIRA_MATH_MODEL_TIER_B") or os.environ.get("MIRA_MATH_MODEL_B", model_default)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    safe_a = model_a.replace("/", "_")
    safe_b = model_b.replace("/", "_")
    run_id = args.run_id or f"{safe_a}_vs_{safe_b}_{timestamp}"
    os.makedirs(args.log_dir, exist_ok=True)
    log_path = os.path.join(args.log_dir, f"MIRA_MATH_{run_id}.log")
    transcript_path = os.path.join(args.log_dir, f"MIRA_MATH_{run_id}.transcript.json")

    logger = logging.getLogger("mira_math")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    logger.handlers = []
    logger.addHandler(logging.StreamHandler())
    logger.addHandler(logging.FileHandler(log_path))
    for h in logger.handlers:
        h.setFormatter(fmt)

    logger.info("Run ID: %s", run_id)
    logger.info("Models: A=%s B=%s", model_a, model_b)
    logger.info("Method: %s", args.method)
    logger.info("Tools: %s", args.tools if args.tools else "None")

    with open(args.instance, "r", encoding="utf-8") as f:
        instance = json.load(f)

    transcript = run_instance(
        instance,
        model_a,
        model_b,
        method=args.method,
        tool_names=args.tools,
        temperature=args.temperature,
        seed=args.seed,
        logger=logger,
    )

    with open(transcript_path, "w", encoding="utf-8") as f:
        json.dump(transcript, f, indent=2)

    logger.info("Transcript saved: %s", transcript_path)


if __name__ == "__main__":
    main()