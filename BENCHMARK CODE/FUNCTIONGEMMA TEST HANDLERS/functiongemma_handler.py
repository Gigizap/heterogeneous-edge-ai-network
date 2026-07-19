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
        inner = ",".join(f"{k}:{_fmt_value(val)}" for k, val in v.items())
        return f"{{{inner}}}"
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
                f"{{{_fmt_args(content)}}}<end_function_response>"
            )
            prev_was_toolish = True

    if not prev_was_toolish:
        prompt += "<start_of_turn>model\n"
    return prompt

def _gbnf_lit(s: str) -> str:
    return s.replace("\\", "\\\\").replace('"', '\\"')

def _build_call_grammar(tools: List[Dict[str, Any]], allow_parallel: bool = True) -> str:
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
            continue
        rbrace = body.rfind("}")
        if rbrace == -1 or rbrace < brace:
            continue
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

    last_role = messages[-1]["role"] if messages else None
    force_call = (
        bool(tools)
        and last_role not in ("tool",)
        and tool_choice not in ("auto", "none")
    )
    if grammar is None and force_call:
        grammar = llama_cpp.LlamaGrammar.from_string(
            _build_call_grammar(tools, allow_parallel=allow_parallel), verbose=False
        )

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
