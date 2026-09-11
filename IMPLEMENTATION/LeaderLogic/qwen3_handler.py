"""
Qwen3 chat handler for llama-cpp-python - SIMPLE version, NO grammar.

Same idea as the functiongemma no-grammar handler, but using Qwen3's native
ChatML + Hermes-style tool-call format instead of Gemma's <start_function_call>
tags. The model generates freely; we parse the result and recover gracefully if
it isn't a valid call.

Why this exists: feeding Qwen3 the Gemma prompt format makes it ignore the tools
and just print the <tool_call> blocks (or prose) as plain text, because the
default llama.cpp chat template for Qwen expects ChatML. This handler builds the
prompt the way Qwen3 was trained and parses the matching output.

Format follows the Qwen3 chat template (tokenizer_config.json):
  * tools advertised in the system turn inside <tools>…</tools> as one JSON
    object per line, each = the full {"type":"function","function":{…}} tool;
  * calls emitted as:
        <tool_call>
        {"name": "fn", "arguments": {...}}
        </tool_call>
    (one block per call, possibly several in a row);
  * tool results sent back inside a user turn:
        <|im_start|>user
        <tool_response>
        ...
        </tool_response><|im_end|>
    consecutive tool messages are merged into a single user turn;
  * <think>…</think> reasoning is stripped from assistant content on parse and
    not re-fed into history (matches the template, which only keeps the reasoning
    of the very last turn).

Trade-off vs. a grammar version:
  + Less code; nothing to keep in sync with tool schemas.
  + Model may freely answer in prose instead of calling a tool.
  - No structural guarantee on output (wrong names / bad types possible).
  (You can still pass grammar= if you build one externally.)

Import this file to register the "qwen3" handler.
"""

import json
import re
from typing import Any, Dict, List, Optional, Union

import llama_cpp
import llama_cpp.llama_types as llama_types
from llama_cpp.llama_chat_format import register_chat_completion_handler

__all__ = ["qwen3_handler"]

IM_START = "<|im_start|>"
IM_END = "<|im_end|>"

TOOLS_PREAMBLE = (
    "# Tools\n\n"
    "You may call one or more functions to assist with the user query.\n\n"
    "You are provided with function signatures within <tools></tools> XML tags:\n"
    "<tools>"
)
TOOLS_POSTAMBLE = (
    "\n</tools>\n\n"
    "For each function call, return a json object with function name and "
    "arguments within <tool_call></tool_call> XML tags:\n"
    "<tool_call>\n"
    '{"name": <function-name>, "arguments": <args-json-object>}\n'
    "</tool_call>"
)


def _tool_json(tool: Dict[str, Any]) -> str:
    """Normalize a tool to the {"type":"function","function":{…}} shape Qwen
    expects, then dump it compact-but-readable (matches `tool | tojson`)."""
    if tool.get("type") == "function" and "function" in tool:
        obj = tool
    else:
        # bare function dict -> wrap it
        obj = {"type": "function", "function": tool}
    return json.dumps(obj, ensure_ascii=False)


def _thinking_from_messages(messages) -> Optional[bool]:
    """Scan messages for the latest /no_think or /think marker.

    Returns False if the last marker is /no_think, True if /think, None if
    neither appears (caller then leaves thinking at the model default). Only a
    standalone token is matched, so words like 'rethink' don't trigger it.
    """
    result: Optional[bool] = None
    for msg in messages:
        text = _stringify(msg.get("content"))
        for m in re.finditer(r"(?<!\w)/(no_think|think)(?!\w)", text):
            result = (m.group(1) == "think")
    return result


def _strip_think(content: str) -> str:
    """Remove a leading/inline <think>…</think> block from assistant content.

    The Qwen template only preserves reasoning for the final turn and strips it
    everywhere else; for history we always drop it so old reasoning doesn't leak
    back into the prompt.
    """
    if not content:
        return content
    if "</think>" in content:
        # keep only what comes after the last </think>
        content = content.split("</think>")[-1].lstrip("\n")
    return content


def _stringify(content: Any) -> str:
    """Coerce a message content (str | list-of-parts | None) into plain text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, (list, tuple)):
        out = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    out.append(item.get("text", ""))
            else:
                out.append(str(item))
        return "".join(out)
    return str(content)


def _build_prompt(messages, tools, enable_thinking: Optional[bool] = None) -> str:
    msgs = list(messages)

    lead_sys = None
    if msgs and msgs[0]["role"] in ("system", "developer"):
        lead_sys = msgs.pop(0)

    prompt = ""

    # ---- system turn (with optional tool advertisement) ----
    if tools:
        prompt += f"{IM_START}system\n"
        if lead_sys is not None:
            sys_text = _stringify(lead_sys.get("content"))
            if sys_text.strip():
                prompt += sys_text.strip() + "\n\n"
        prompt += TOOLS_PREAMBLE
        for tool in tools:
            prompt += "\n" + _tool_json(tool)
        prompt += TOOLS_POSTAMBLE
        prompt += f"{IM_END}\n"
    elif lead_sys is not None:
        sys_text = _stringify(lead_sys.get("content"))
        if sys_text.strip():
            prompt += f"{IM_START}system\n{sys_text.strip()}{IM_END}\n"

    # ---- conversation ----
    i = 0
    n = len(msgs)
    while i < n:
        msg = msgs[i]
        role = msg["role"]

        if role in ("system", "developer"):
            text = _stringify(msg.get("content")).strip()
            if text:
                prompt += f"{IM_START}system\n{text}{IM_END}\n"
            i += 1

        elif role == "user":
            text = _stringify(msg.get("content")).strip()
            prompt += f"{IM_START}user\n{text}{IM_END}\n"
            i += 1

        elif role == "assistant":
            content = _strip_think(_stringify(msg.get("content"))).strip()
            tool_calls = msg.get("tool_calls") or []
            prompt += f"{IM_START}assistant\n{content}"
            for idx, tc in enumerate(tool_calls):
                fn = tc["function"] if "function" in tc else tc
                args = fn.get("arguments", {})
                if not isinstance(args, str):
                    args = json.dumps(args, ensure_ascii=False)
                # newline before each call block (and between content + first call)
                if content or idx > 0:
                    prompt += "\n"
                prompt += (
                    "<tool_call>\n"
                    f'{{"name": "{fn["name"]}", "arguments": {args}}}\n'
                    "</tool_call>"
                )
            prompt += f"{IM_END}\n"
            i += 1

        elif role == "tool":
            # merge consecutive tool messages into one user turn
            prompt += f"{IM_START}user"
            while i < n and msgs[i]["role"] == "tool":
                prompt += (
                    "\n<tool_response>\n"
                    f"{_stringify(msgs[i].get('content'))}\n"
                    "</tool_response>"
                )
                i += 1
            prompt += f"{IM_END}\n"

        else:
            i += 1

    prompt += f"{IM_START}assistant\n"
    if enable_thinking is False:
        prompt += "<think>\n\n</think>\n\n"
    return prompt


# --------------------------------------------------------------------------- #
# Output parsing
#
# Qwen emits each call as a <tool_call>…</tool_call> block whose body is JSON:
#     <tool_call>
#     {"name": "fn", "arguments": {"a": 1}}
#     </tool_call>
# We extract every such block and json-load it. We're tolerant of:
#   * a missing closing </tool_call> on the LAST block (hit max_tokens) - we
#     fall back to brace-matching to recover the JSON object;
#   * a leaked <think>…</think> preceding the calls (stripped for content).
# A block whose JSON can't be parsed is skipped rather than mis-emitted.
# --------------------------------------------------------------------------- #
_TOOLCALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)


def _extract_first_json_object(s: str) -> Optional[str]:
    """Return the first balanced {...} JSON object substring, or None.

    Brace-aware and string/escape-aware so braces inside string values don't
    throw off the depth count. Used to rescue an unterminated final tool_call.
    """
    start = s.find("{")
    if start == -1:
        return None
    depth, in_str, esc = 0, False, False
    for j in range(start, len(s)):
        c = s[j]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
        else:
            if c == '"':
                in_str = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    return s[start : j + 1]
    return None


def _parse_calls(text: str) -> List[Dict[str, str]]:
    calls: List[Dict[str, str]] = []

    # 1) well-formed, closed blocks
    bodies = _TOOLCALL_RE.findall(text)

    # 2) rescue an unterminated trailing <tool_call> (no closing tag)
    last_open = text.rfind("<tool_call>")
    if last_open != -1:
        tail = text[last_open + len("<tool_call>"):]
        if "</tool_call>" not in tail:
            obj = _extract_first_json_object(tail)
            if obj is not None:
                bodies.append(obj)

    for body in bodies:
        body = body.strip()
        if not body:
            continue
        try:
            obj = json.loads(body)
        except json.JSONDecodeError:
            # last-ditch: pull a balanced object out of noisy text
            rescued = _extract_first_json_object(body)
            if rescued is None:
                continue
            try:
                obj = json.loads(rescued)
            except json.JSONDecodeError:
                continue
        name = obj.get("name")
        if not name:
            continue
        args = obj.get("arguments", {})
        if isinstance(args, str):
            # arguments may already be a JSON string; keep it if valid, else wrap
            try:
                json.loads(args)
                args_str = args
            except json.JSONDecodeError:
                args_str = json.dumps({"value": args}, ensure_ascii=False)
        else:
            args_str = json.dumps(args, ensure_ascii=False)
        calls.append({"name": name, "arguments": args_str})

    return calls


@register_chat_completion_handler("qwen3")
def qwen3_handler(
    llama: llama_cpp.Llama,
    messages: List[llama_types.ChatCompletionRequestMessage],
    functions: Optional[List[llama_types.ChatCompletionFunction]] = None,
    function_call: Optional[llama_types.ChatCompletionRequestFunctionCall] = None,
    tools: Optional[List[llama_types.ChatCompletionTool]] = None,
    tool_choice: Optional[llama_types.ChatCompletionToolChoiceOption] = None,
    temperature: float = 0.7,
    top_p: float = 0.8,
    top_k: int = 20,
    min_p: float = 0.0,
    stream: bool = False,
    stop: Optional[Union[str, List[str]]] = None,
    max_tokens: Optional[int] = 512,
    repeat_penalty: float = 1.05,
    model: Optional[str] = None,
    logits_processor: Optional[llama_cpp.LogitsProcessorList] = None,
    grammar: Optional[llama_cpp.LlamaGrammar] = None,
    enable_thinking: Optional[bool] = None,
    **kwargs: Any,
) -> llama_types.CreateChatCompletionResponse:
    if functions and not tools:
        tools = [{"type": "function", "function": f} for f in functions]

    # llama-cpp-python's create_chat_completion validates kwargs and won't
    # forward an unknown `enable_thinking` here, so we also honor a /no_think
    # (or /think) marker placed in any message's text. Explicit kwarg wins;
    # otherwise the LAST marker found in the conversation decides, matching
    # Qwen's own "latest marker" semantics.
    if enable_thinking is None:
        enable_thinking = _thinking_from_messages(messages)

    prompt = _build_prompt(messages, tools, enable_thinking=enable_thinking)

    # Stop on <|im_end|> (turn end). We do NOT stop on </tool_call> so the model
    # can emit several parallel calls in one turn; the runaway risk that the
    # gemma handler guards against isn't the failure mode here.
    stop_tokens = [stop] if isinstance(stop, str) else list(stop or [])
    if IM_END not in stop_tokens:
        stop_tokens.append(IM_END)

    completion = llama.create_completion(
        prompt=prompt,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        min_p=min_p,
        stream=False,
        stop=stop_tokens,
        max_tokens=max_tokens,
        repeat_penalty=repeat_penalty,
        model=model,
        logits_processor=logits_processor,
        grammar=grammar,  # only used if explicitly passed in
    )

    text = completion["choices"][0]["text"]
    calls = _parse_calls(text) if tools else []

    if calls:
        message: Dict[str, Any] = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": f"call_{c['name']}_{completion['id']}_{idx}",
                    "type": "function",
                    "function": {"name": c["name"], "arguments": c["arguments"]},
                }
                for idx, c in enumerate(calls)
            ],
        }
        finish = "tool_calls"
    else:
        # strip any reasoning block from the visible content
        message = {"role": "assistant", "content": _strip_think(text).strip()}
        finish = "stop"

    message["_raw_text"] = text

    return {
        "id": "chat" + completion["id"],
        "object": "chat.completion",
        "created": completion["created"],
        "model": completion["model"],
        "choices": [
            {"index": 0, "message": message, "logprobs": None, "finish_reason": finish}
        ],
        "usage": completion["usage"],
    }