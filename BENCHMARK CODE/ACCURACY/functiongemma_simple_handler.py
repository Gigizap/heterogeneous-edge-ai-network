"""
FunctionGemma chat handler for llama-cpp-python — SIMPLE version, NO grammar.

Same spec-verified prompt format and parsing as functiongemma_handler.py, but
without GBNF grammar generation/enforcement. The model generates freely; we
parse the result and recover gracefully if it isn't a valid call.

Format verified against:
  https://ai.google.dev/gemma/docs/functiongemma/formatting-and-best-practices
  https://ai.google.dev/gemma/docs/core/prompt-structure

Trade-off vs. the grammar version:
  + Less code; nothing to keep in sync with tool schemas.
  + Model may freely answer in prose instead of calling a tool.
  - No structural guarantee on the output (wrong names / bad types possible).
  (It is still possible to pass grammar= if built externally.)

Import EITHER this file OR functiongemma_handler.py, not both.
"""

import json
from typing import Any, Dict, List, Optional, Union

import llama_cpp
import llama_cpp.llama_types as llama_types
from llama_cpp.llama_chat_format import register_chat_completion_handler

__all__ = ["functiongemma_handler"]

TRIGGER = "You are a model that can do function calling with the following functions"

_FG_TYPES = {
    "string": "STRING", "integer": "NUMBER", "number": "NUMBER",
    "boolean": "BOOLEAN", "object": "OBJECT", "array": "ARRAY",
}


def _escape(s: str) -> str:
    return f"<escape>{s}<escape>"


def _fmt_value(v: Any) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return str(v)
    if isinstance(v, dict):
        return "{" + ",".join(f"{k}:{_fmt_value(val)}" for k, val in v.items()) + "}"
    if isinstance(v, (list, tuple)):
        return "[" + ",".join(_fmt_value(x) for x in v) + "]"
    return _escape(str(v))


def _fmt_args(arguments: Union[str, Dict[str, Any]]) -> str:
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            return arguments
    return ",".join(f"{k}:{_fmt_value(v)}" for k, v in arguments.items())


def _fmt_tool_response(content: Any) -> str:
    """Render the body of a <start_function_response>…<end_function_response>.

    Tool results in the wild are not always a flat {key: value} object:
      * a dict   -> rendered as key:value pairs (the spec's happy path)
      * a list   -> e.g. [{"from": "cam", "name": "Unknown", "confidence": 0.31}];
                    common for sensor/array results. Rendered as a value:[...] list.
      * a scalar -> wrapped as value:<scalar>.
    A JSON string is parsed first; anything that fails to parse is escaped as-is.

    This replaces the old `_fmt_args(content)` call, which assumed a dict and
    crashed with `'list' object has no attribute 'items'` on list payloads.
    """
    parsed: Any = content
    if isinstance(content, str):
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            return _escape(content)

    if isinstance(parsed, dict):
        return ",".join(f"{k}:{_fmt_value(v)}" for k, v in parsed.items())
    if isinstance(parsed, (list, tuple)):
        return f"value:{_fmt_value(list(parsed))}"
    return f"value:{_fmt_value(parsed)}"


def _fg_type(t: str) -> str:
    return _FG_TYPES.get(t, "STRING")


def _param_schema(properties: Dict[str, Any], required: List[str]) -> str:
    parts = []
    for name, defn in properties.items():
        body = f"{name}:{{"
        if defn.get("description"):
            body += f"description:{_escape(defn['description'])},"
        if defn.get("enum"):
            body += "enum:[" + ",".join(_escape(str(v)) for v in defn["enum"]) + "],"
        if defn.get("type") == "object" and "properties" in defn:
            body += f"properties:{{{_param_schema(defn['properties'], defn.get('required', []))}}},"
        if defn.get("type") == "array" and "items" in defn:
            body += f"items:{{type:{_escape(_fg_type(defn['items'].get('type', 'string')))}}},"
        body += f"type:{_escape(_fg_type(defn.get('type', 'string')))}}}"
        parts.append(body)
    out = ""
    if parts:
        out += f"properties:{{{','.join(parts)}}},"
    if required:
        out += f"required:[{','.join(_escape(r) for r in required)}],"
    out += f"type:{_escape('OBJECT')}"
    return out


def _declaration(tool: Dict[str, Any]) -> str:
    func = tool["function"] if tool.get("type") == "function" else tool
    params = func.get("parameters", {}) or {}
    decl = f"declaration:{func['name']}{{description:{_escape(func.get('description', ''))}"
    if params:
        decl += f",parameters:{{{_param_schema(params.get('properties', {}), params.get('required', []))}}}"
    decl += "}"
    return f"<start_function_declaration>{decl}<end_function_declaration>"


def _build_prompt(messages, tools) -> str:
    prompt = ""
    msgs = list(messages)
    lead_sys = None
    if msgs and msgs[0]["role"] in ("system", "developer"):
        lead_sys = msgs.pop(0)

    if tools or lead_sys:
        prompt += "<start_of_turn>developer\n"
        if lead_sys is not None and lead_sys.get("content"):
            prompt += str(lead_sys["content"]).strip()
        else:
            prompt += TRIGGER
        if tools:
            prompt += "".join(_declaration(t) for t in tools)
        prompt += "<end_of_turn>\n"

    prev_was_toolish = False
    for msg in msgs:
        role = msg["role"]
        content = msg.get("content")
        if role in ("system", "developer"):
            prompt += f"<start_of_turn>developer\n{str(content).strip()}<end_of_turn>\n"
            prev_was_toolish = False
        elif role == "user":
            prompt += f"<start_of_turn>user\n{str(content).strip()}<end_of_turn>\n"
            prev_was_toolish = False
        elif role == "assistant":
            tool_calls = msg.get("tool_calls") or []
            if content:
                prompt += f"<start_of_turn>model\n{str(content).strip()}<end_of_turn>\n"
            if tool_calls:
                prompt += "<start_of_turn>model\n"
                for tc in tool_calls:
                    fn = tc["function"]
                    prompt += (
                        f"<start_function_call>call:{fn['name']}"
                        f"{{{_fmt_args(fn['arguments'])}}}<end_function_call>"
                    )
                prev_was_toolish = True
        elif role == "tool":
            name = msg.get("name", "")
            prompt += (
                f"<start_function_response>response:{name}"
                f"{{{_fmt_tool_response(content)}}}<end_function_response>"
            )
            prev_was_toolish = True

    if not prev_was_toolish:
        prompt += "<start_of_turn>model\n"
    return prompt


# --------------------------------------------------------------------------- #
# Output parsing
#
# Ported verbatim from functiongemma_handler.py so the two handlers parse
# identically. This rfind-based parser is robust to the missing-stop-token
# case: it locates the LAST closing brace, so a complete-but-unterminated call
# (no trailing <end_function_call> because generation hit max_tokens) still
# parses, and nested-brace arguments aren't truncated. A genuinely truncated
# fragment with no closing brace is skipped rather than mis-parsed.
#
# This replaces the previous regex-based parser (_CALL_RE / _ARG_RE / _cast),
# whose non-greedy `\{(.*?)\}` truncated nested-brace arguments.
# --------------------------------------------------------------------------- #
def _split_args(args_str: str) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    if not args_str.strip():
        return result
    parts, current, depth, in_escape, i = [], "", 0, False, 0
    while i < len(args_str):
        if args_str[i:].startswith("<escape>"):
            in_escape = not in_escape
            current += "<escape>"
            i += 8
            continue
        c = args_str[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
        elif c == "," and depth == 0 and not in_escape:
            parts.append(current)
            current, i = "", i + 1
            continue
        current += c
        i += 1
    if current.strip():
        parts.append(current)

    for part in parts:
        if ":" not in part:
            continue
        key, raw = part.split(":", 1)
        key, raw = key.strip(), raw.strip()
        if raw.startswith("<escape>"):
            val: Any = raw.replace("<escape>", "")
        elif raw in ("true", "false"):
            val = raw == "true"
        else:
            try:
                num = float(raw)
                val = int(num) if num.is_integer() else num
            except ValueError:
                val = raw.replace("<escape>", "")
        result[key] = val
    return result


def _parse_calls(text: str) -> List[Dict[str, str]]:
    """Extract every complete call:name{...} block.

    Robust to the 270M failure mode where the model emits the FIRST call wrapped
    in <start_function_call> but the runaway repeats as bare `call:...` without a
    new opening tag. We split on the closing tag, strip any opening tag, and
    require a complete `{...}` so a truncated trailing fragment is discarded.
    """
    end = "<end_function_call>"
    calls: List[Dict[str, str]] = []
    for seg in text.split(end):
        seg = seg.replace("<start_function_call>", "")
        idx = seg.find("call:")
        if idx == -1:
            continue
        body = seg[idx + len("call:"):]
        brace = body.find("{")
        if brace == -1:
            continue  # no args block started -> incomplete, skip
        rbrace = body.rfind("}")
        if rbrace == -1 or rbrace < brace:
            continue  # closing brace missing -> truncated fragment, skip
        name = body[:brace].strip()
        if not name:
            continue
        args = _split_args(body[brace + 1:rbrace])
        calls.append({"name": name, "arguments": json.dumps(args)})
    return calls


@register_chat_completion_handler("functiongemma")
def functiongemma_handler(
    llama: llama_cpp.Llama,
    messages: List[llama_types.ChatCompletionRequestMessage],
    functions: Optional[List[llama_types.ChatCompletionFunction]] = None,
    function_call: Optional[llama_types.ChatCompletionRequestFunctionCall] = None,
    tools: Optional[List[llama_types.ChatCompletionTool]] = None,
    tool_choice: Optional[llama_types.ChatCompletionToolChoiceOption] = None,
    temperature: float = 0.2,
    top_p: float = 0.95,
    top_k: int = 64,
    min_p: float = 0.0,
    stream: bool = False,
    stop: Optional[Union[str, List[str]]] = None,
    max_tokens: Optional[int] = 256,
    repeat_penalty: float = 1.1,
    model: Optional[str] = None,
    logits_processor: Optional[llama_cpp.LogitsProcessorList] = None,
    grammar: Optional[llama_cpp.LlamaGrammar] = None,
    **kwargs: Any,
) -> llama_types.CreateChatCompletionResponse:
    if functions and not tools:
        tools = [{"type": "function", "function": f} for f in functions]

    prompt = _build_prompt(messages, tools)

    stop_tokens = [stop] if isinstance(stop, str) else list(stop or [])
    # No grammar here to force a stop, so we stop on <end_function_call>: the
    # model halts right after the first complete call, which kills the 270M
    # runaway-repeat at generation time (saving wasted tokens). The trade-off is
    # this disables parallel calls on the no-grammar path -- use the grammar
    # handler with allow_parallel=True if parallel is needed (broken).
    for s in ("<end_of_turn>", "<end_function_call>", "<start_function_response>"):
        if s not in stop_tokens:
            stop_tokens.append(s)

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
        grammar=grammar,  # only used if explicitely passed in
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
        message = {"role": "assistant", "content": text.strip()}
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