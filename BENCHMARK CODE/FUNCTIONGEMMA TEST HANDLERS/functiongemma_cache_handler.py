"""
FunctionGemma chat handler for llama-cpp-python — WITH grammar-constrained decoding.

Format verified against Google's official spec:
  https://ai.google.dev/gemma/docs/functiongemma/formatting-and-best-practices
  https://ai.google.dev/gemma/docs/core/prompt-structure
and cross-checked against the chat_template embedded in the Unsloth GGUF.

Key format facts this handler obeys:
  * Tool definitions go in a single leading `developer` turn, prefixed with the
    mandatory trigger phrase, each tool wrapped in
    <start_function_declaration>declaration:NAME{...}<end_function_declaration>.
  * A call is `<start_function_call>call:NAME{key:value,...}<end_function_call>`
    where keys are bare and STRING values are wrapped in <escape>...<escape>;
    numbers and booleans are bare.
  * A tool RESULT is emitted bare (NO <start_of_turn> wrapper, NO trailing
    <end_of_turn>) as
    <start_function_response>response:NAME{key:value,...}<end_function_response>.
  * <start_function_response> is an additional stop sequence.
  * The model is trained ONLY for single-turn and PARALLEL calls — not
    multi-step chaining. Parallel calls => several <end_function_call> blocks
    in one model turn, so we must NOT stop on <end_function_call>.

Usage:
    import llama_cpp
    import functiongemma_handler  # noqa: F401  (registers "functiongemma_cache")

    llm = llama_cpp.Llama(
        model_path="functiongemma-270m-it-Q4_K_M.gguf",
        chat_format="functiongemma",
        n_ctx=2048,
    )
    resp = llm.create_chat_completion(messages=[...], tools=[...])

Import EITHER this file OR functiongemma_handler_simple.py, not both.
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


# --------------------------------------------------------------------------- #
# Value formatting (spec: strings escaped, numbers/bools bare)
# --------------------------------------------------------------------------- #
def _escape(s: str) -> str:
    return f"<escape>{s}<escape>"


def _fmt_value(v: Any) -> str:
    """Render a Python value into FunctionGemma's call/response value syntax."""
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return str(v)
    if isinstance(v, dict):
        inner = ",".join(f"{k}:{_fmt_value(val)}" for k, val in v.items())
        return f"{{{inner}}}"
    if isinstance(v, (list, tuple)):
        return "[" + ",".join(_fmt_value(x) for x in v) + "]"
    return _escape(str(v))


def _fmt_args(arguments: Union[str, Dict[str, Any]]) -> str:
    """Render call/response arguments. Keys are bare; values per _fmt_value."""
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            return arguments  # already FG-formatted; pass through
    return ",".join(f"{k}:{_fmt_value(v)}" for k, v in arguments.items())


# --------------------------------------------------------------------------- #
# Tool declaration block
# --------------------------------------------------------------------------- #
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
    decl = (
        f"declaration:{func['name']}{{"
        f"description:{_escape(func.get('description', ''))}"
    )
    if params:
        decl += f",parameters:{{{_param_schema(params.get('properties', {}), params.get('required', []))}}}"
    decl += "}"
    return f"<start_function_declaration>{decl}<end_function_declaration>"


# --------------------------------------------------------------------------- #
# Prompt construction
# --------------------------------------------------------------------------- #
def _build_prompt(messages, tools) -> str:
    prompt = ""
    msgs = list(messages)

    # Leading developer turn: system/developer content and/or tool declarations.
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

    prev_was_toolish = False  # tracks tool_call / tool_response (no <end_of_turn>)
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
                # No <end_of_turn> after a tool_call turn (matches template).
                prev_was_toolish = True

        elif role == "tool":
            # Tool result: emitted bare, no <start_of_turn>, no <end_of_turn>.
            name = msg.get("name", "")
            prompt += (
                f"<start_function_response>response:{name}"
                f"{{{_fmt_args(content)}}}<end_function_response>"
            )
            prev_was_toolish = True

    # Generation prompt. The template omits the leading <start_of_turn>model
    # if the previous turn was a tool_response (model continues directly).
    if not prev_was_toolish:
        prompt += "<start_of_turn>model\n"
    return prompt


# --------------------------------------------------------------------------- #
# GBNF grammar generation
# --------------------------------------------------------------------------- #
def _gbnf_lit(s: str) -> str:
    return s.replace("\\", "\\\\").replace('"', '\\"')


def _build_call_grammar(tools: List[Dict[str, Any]], allow_parallel: bool = True) -> str:
    """GBNF matching one OR a sequence of valid FunctionGemma calls."""
    lines: List[str] = []
    tool_refs: List[str] = []
    need = {"str": False, "num": False, "bool": False}

    for i, tool in enumerate(tools):
        func = tool["function"] if tool.get("type") == "function" else tool
        name = _gbnf_lit(func["name"])
        props = (func.get("parameters", {}) or {}).get("properties", {}) or {}
        rule = f"tool{i}"

        if not props:
            lines.append(f'{rule} ::= "call:{name}{{}}"')
            tool_refs.append(rule)
            continue

        pair_alts = []
        for pname, pdef in props.items():
            key = _gbnf_lit(pname)
            if pdef.get("enum"):
                opts = " | ".join(f'"<escape>{_gbnf_lit(str(o))}<escape>"' for o in pdef["enum"])
                pair_alts.append(f'"{key}:" ( {opts} )')
            else:
                t = pdef.get("type", "string")
                if t in ("integer", "number"):
                    need["num"] = True
                    pair_alts.append(f'"{key}:" numval')
                elif t == "boolean":
                    need["bool"] = True
                    pair_alts.append(f'"{key}:" boolval')
                else:
                    need["str"] = True
                    pair_alts.append(f'"{key}:" strval')

        pair = f"pair{i}"
        lines.append(f"{pair} ::= " + " | ".join(pair_alts))
        lines.append(f'{rule} ::= "call:{name}{{" ( {pair} ( "," {pair} )* )? "}}"')
        tool_refs.append(rule)

    one = '"<start_function_call>" ( ' + " | ".join(tool_refs) + ' ) "<end_function_call>"'
    if allow_parallel:
        lines.insert(0, f"call ::= {one}")
        root = "root ::= call+"
    else:
        root = f"root ::= {one}"

    out = [root] + lines
    if need["str"]:
        out += ['strval ::= "<escape>" strchar* "<escape>"', "strchar ::= [^<]"]
    if need["num"]:
        out.append('numval ::= "-"? [0-9]+ ( "." [0-9]+ )?')
    if need["bool"]:
        out.append('boolval ::= "true" | "false"')
    return "\n".join(out)


# --------------------------------------------------------------------------- #
# Compiled-grammar cache
# --------------------------------------------------------------------------- #
# Compiling a LlamaGrammar from a GBNF string on every request is the dominant
# per-call overhead. The grammar depends ONLY on the tool-set and allow_parallel,
# so we key a cache on a canonical signature of those and reuse the compiled
# LlamaGrammar object across requests. Bounded so a churning tool-set can't grow
# it without limit.
_GRAMMAR_CACHE: "Dict[Any, llama_cpp.LlamaGrammar]" = {}
_GRAMMAR_CACHE_MAX = 128


def _grammar_cache_key(tools: List[Dict[str, Any]], allow_parallel: bool) -> Any:
    """Canonical, hashable signature of everything _build_call_grammar reads.

    _build_call_grammar only looks at each tool's name and its parameter
    properties (per-prop: enum, type), plus allow_parallel. We capture exactly
    those, in order, so two tool-sets that produce identical GBNF share an entry
    while any schema change yields a distinct key.
    """
    sig: List[Any] = [allow_parallel]
    for tool in tools:
        func = tool["function"] if tool.get("type") == "function" else tool
        props = (func.get("parameters", {}) or {}).get("properties", {}) or {}
        prop_sig = tuple(
            (
                pname,
                pdef.get("type", "string"),
                tuple(str(o) for o in pdef["enum"]) if pdef.get("enum") else None,
            )
            for pname, pdef in props.items()
        )
        sig.append((func["name"], prop_sig))
    return tuple(sig)


def _get_cached_grammar(
    tools: List[Dict[str, Any]], allow_parallel: bool
) -> llama_cpp.LlamaGrammar:
    """Return a compiled LlamaGrammar for this tool-set, building once and caching."""
    key = _grammar_cache_key(tools, allow_parallel)
    grammar = _GRAMMAR_CACHE.get(key)
    if grammar is None:
        grammar = llama_cpp.LlamaGrammar.from_string(
            _build_call_grammar(tools, allow_parallel=allow_parallel), verbose=False
        )
        if len(_GRAMMAR_CACHE) >= _GRAMMAR_CACHE_MAX:
            # Simple bound: drop the oldest inserted entry.
            _GRAMMAR_CACHE.pop(next(iter(_GRAMMAR_CACHE)))
        _GRAMMAR_CACHE[key] = grammar
    return grammar


# --------------------------------------------------------------------------- #
# Output parsing
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


# --------------------------------------------------------------------------- #
# Handler
# --------------------------------------------------------------------------- #
@register_chat_completion_handler("functiongemma_cache")
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
    typical_p: float = 1.0,
    stream: bool = False,
    stop: Optional[Union[str, List[str]]] = None,
    max_tokens: Optional[int] = 256,
    presence_penalty: float = 0.0,
    frequency_penalty: float = 0.0,
    repeat_penalty: float = 1.1,
    model: Optional[str] = None,
    logits_processor: Optional[llama_cpp.LogitsProcessorList] = None,
    grammar: Optional[llama_cpp.LlamaGrammar] = None,
    allow_parallel: bool = False,
    **kwargs: Any,
) -> llama_types.CreateChatCompletionResponse:
    if functions and not tools:
        tools = [{"type": "function", "function": f} for f in functions]

    prompt = _build_prompt(messages, tools)

    # Force a structurally-valid call when a fresh user request needs tools.
    # Skip after a tool result (model should answer in prose) and when
    # tool_choice explicitly defers/forbids.
    last_role = messages[-1]["role"] if messages else None
    force_call = (
        bool(tools)
        and last_role not in ("tool",)
        and tool_choice not in ("auto", "none")
    )
    if grammar is None and force_call:
        # Default: single-call grammar. Once one <start_function_call>...
        # <end_function_call> completes, the root rule is satisfied and the only
        # legal next token is end-of-generation -> the model is forced to STOP.
        # This is what kills the 270M runaway-repeat at the source. Opt into
        # allow_parallel=True only if you actually need parallel calls and accept
        # that this model may over-generate.
        #
        # Compiling the GBNF is the dominant per-request cost, and the grammar
        # depends only on the tool-set + allow_parallel, so reuse a cached
        # compiled grammar keyed on that signature instead of rebuilding here.
        grammar = _get_cached_grammar(tools, allow_parallel)

    # Stop tokens. <start_function_response> per spec. We do NOT stop on
    # <end_function_call> so parallel calls can complete.
    stop_tokens = [stop] if isinstance(stop, str) else list(stop or [])
    for s in ("<end_of_turn>", "<end_function_call>", "<start_function_response>"):
        if s not in stop_tokens:
            stop_tokens.append(s)

    completion = llama.create_completion(
        prompt=prompt,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        min_p=min_p,
        typical_p=typical_p,
        stream=False,
        stop=stop_tokens,
        max_tokens=max_tokens,
        presence_penalty=presence_penalty,
        frequency_penalty=frequency_penalty,
        repeat_penalty=repeat_penalty,
        model=model,
        logits_processor=logits_processor,
        grammar=grammar,
    )

    text = completion["choices"][0]["text"]
    calls = _parse_calls(text) if tools else []
    # In single-call mode keep only the first call. Even if the no-grammar path
    # over-generated, the user gets one clean call instead of a runaway list.
    if calls and not allow_parallel:
        calls = calls[:1]

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