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
    if tool.get("type") == "function" and "function" in tool:
        obj = tool
    else:
        obj = {"type": "function", "function": tool}
    return json.dumps(obj, ensure_ascii=False)

def _thinking_from_messages(messages) -> Optional[bool]:
    result: Optional[bool] = None
    for msg in messages:
        text = _stringify(msg.get("content"))
        for m in re.finditer(r"(?<!\w)/(no_think|think)(?!\w)", text):
            result = (m.group(1) == "think")
    return result

def _strip_think(content: str) -> str:
    if not content:
        return content
    if "</think>" in content:
        content = content.split("</think>")[-1].lstrip("\n")
    return content

def _stringify(content: Any) -> str:
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

_TOOLCALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)

def _extract_first_json_object(s: str) -> Optional[str]:
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

    bodies = _TOOLCALL_RE.findall(text)

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

    if enable_thinking is None:
        enable_thinking = _thinking_from_messages(messages)

    prompt = _build_prompt(messages, tools, enable_thinking=enable_thinking)

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
        grammar=grammar,
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
