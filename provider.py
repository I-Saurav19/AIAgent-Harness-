"""LLM provider adapters with a shared chat interface."""

import json
from abc import ABC, abstractmethod
from config import get_api_key


class LLMProvider(ABC):
    @abstractmethod
    def chat(self, messages, tools, model):
        """Return text and normalized tool calls."""

    def stream(self, messages, tools, model):
        """Yield normalized text, tool-call, and completion events."""
        response = self.chat(messages=messages, tools=tools, model=model)
        if response.get("text"):
            yield {"type": "text_delta", "text": response["text"]}
        for call in response.get("tool_calls", []):
            yield {"type": "tool_call", "tool_call": call}
        yield {"type": "completed"}


def _openai_tools(tools):
    return [{"type": "function", "function": {"name": t["name"], "description": t["description"], "parameters": t["parameters"]}} for t in tools]


def _normalize_chat_response(response):
    message = response.choices[0].message
    return {"text": message.content or "", "tool_calls": [
        {"id": call.id, "name": call.function.name, "arguments": json.loads(call.function.arguments or "{}")}
        for call in (message.tool_calls or [])
    ]}


def _stream_chat_completion(client, messages, tools, model):
    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        tools=_openai_tools(tools),
        tool_choice="auto",
        stream=True,
    )
    pending_calls = {}
    for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        if delta.content:
            yield {"type": "text_delta", "text": delta.content}
        for fragment in delta.tool_calls or []:
            call = pending_calls.setdefault(fragment.index, {"id": "", "name": "", "arguments_json": ""})
            if fragment.id:
                call["id"] = fragment.id
            if fragment.function:
                if fragment.function.name:
                    call["name"] += fragment.function.name
                if fragment.function.arguments:
                    call["arguments_json"] += fragment.function.arguments

    for index, call in sorted(pending_calls.items()):
        try:
            arguments = json.loads(call["arguments_json"])
        except json.JSONDecodeError as error:
            yield {
                "type": "tool_call_error",
                "name": call["name"] or f"tool-{index}",
                "error": f"Tool arguments were not valid JSON: {error.msg}.",
            }
            continue
        if not isinstance(arguments, dict):
            yield {
                "type": "tool_call_error",
                "name": call["name"] or f"tool-{index}",
                "error": "Tool arguments must be a JSON object.",
            }
            continue
        yield {
            "type": "tool_call",
            "tool_call": {
                "id": call["id"] or f"tool-{index}",
                "name": call["name"],
                "arguments": arguments,
            },
        }
    yield {"type": "completed"}


class OpenAIProvider(LLMProvider):
    def __init__(self):
        from openai import OpenAI
        self.client = OpenAI(api_key=get_api_key("openai"))

    def chat(self, messages, tools, model):
        response = self.client.chat.completions.create(model=model, messages=messages, tools=_openai_tools(tools), tool_choice="auto")
        return _normalize_chat_response(response)

    def stream(self, messages, tools, model):
        yield from _stream_chat_completion(self.client, messages, tools, model)


class GroqProvider(LLMProvider):
    def __init__(self):
        from groq import Groq
        self.client = Groq(api_key=get_api_key("groq"))

    def chat(self, messages, tools, model):
        response = self.client.chat.completions.create(model=model, messages=messages, tools=_openai_tools(tools), tool_choice="auto")
        return _normalize_chat_response(response)

    def stream(self, messages, tools, model):
        try:
            yield from _stream_chat_completion(self.client, messages, tools, model)
        except Exception as error:
            message = str(error)
            if "failed to parse tool call arguments as json" not in message.lower():
                raise
            yield {
                "type": "tool_call_error",
                "name": "unknown tool",
                "error": "Groq rejected malformed streamed tool-call arguments as JSON.",
            }
            yield {"type": "completed"}


class XAIProvider(LLMProvider):
    """xAI's OpenAI-compatible chat completions API."""

    def __init__(self):
        from openai import OpenAI
        self.client = OpenAI(api_key=get_api_key("xai"), base_url="https://api.x.ai/v1")

    def chat(self, messages, tools, model):
        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            tools=_openai_tools(tools),
            tool_choice="auto",
        )
        return _normalize_chat_response(response)

    def stream(self, messages, tools, model):
        yield from _stream_chat_completion(self.client, messages, tools, model)


class GeminiProvider(LLMProvider):
    def __init__(self):
        from google import genai
        self.client = genai.Client(api_key=get_api_key("gemini"))
        self._call_counter = 0
        self._function_call_contents = {}
        self._function_response_ids = {}

    def _remember_function_call(self, content, function_call):
        self._call_counter += 1
        call_id = f"gemini-{self._call_counter}"
        self._function_call_contents[call_id] = content
        self._function_response_ids[call_id] = function_call.id
        return {
            "id": call_id,
            "name": function_call.name,
            "arguments": dict(function_call.args or {}),
        }

    def _request(self, messages, tools):
        from google.genai import types
        system_instruction = next((m["content"] for m in messages if m["role"] == "system"), None)
        contents = []
        index = 0
        while index < len(messages):
            message = messages[index]
            role = message["role"]
            if role == "system":
                index += 1
                continue
            elif role == "user":
                contents.append(types.Content(role="user", parts=[types.Part(text=message["content"])]))
            elif role == "assistant":
                calls = message.get("tool_calls", [])
                if calls:
                    original_content = self._function_call_contents.get(calls[0]["id"])
                    if original_content is None or any(
                        call["id"] not in self._function_call_contents
                        or self._function_call_contents[call["id"]] is not original_content
                        for call in calls
                    ):
                        raise ValueError("Gemini function-call history is missing its original response metadata.")
                    contents.append(original_content)
                elif message.get("content"):
                    contents.append(types.Content(role="model", parts=[types.Part(text=message["content"])]))
            elif role == "tool":
                response_parts = []
                while index < len(messages) and messages[index]["role"] == "tool":
                    tool_message = messages[index]
                    try:
                        result = json.loads(tool_message["content"])
                    except (json.JSONDecodeError, TypeError):
                        result = tool_message["content"]
                    call_id = tool_message.get("tool_call_id")
                    function_response = types.FunctionResponse(
                        name=tool_message.get("name", "tool"),
                        response={"result": result},
                        id=self._function_response_ids.get(call_id),
                    )
                    response_parts.append(types.Part(function_response=function_response))
                    index += 1
                # Gemini represents function results as user content containing
                # FunctionResponse parts. Keep each response's name and call id
                # so it remains associated with the corresponding model call.
                contents.append(types.Content(role="user", parts=response_parts))
                continue
            index += 1

        declarations = [types.FunctionDeclaration(name=t["name"], description=t["description"], parameters_json_schema=t["parameters"]) for t in tools]
        config = types.GenerateContentConfig(system_instruction=system_instruction, tools=[types.Tool(function_declarations=declarations)])
        return contents, config

    def chat(self, messages, tools, model):
        contents, config = self._request(messages, tools)
        response = self.client.models.generate_content(
            model=model,
            contents=contents,
            config=config,
        )
        text_parts, tool_calls = [], []
        candidate = response.candidates[0] if response.candidates else None
        if candidate and candidate.content:
            for index, part in enumerate(candidate.content.parts or []):
                if part.text:
                    text_parts.append(part.text)
                if part.function_call:
                    tool_calls.append(self._remember_function_call(candidate.content, part.function_call))
        return {"text": "".join(text_parts), "tool_calls": tool_calls}

    def stream(self, messages, tools, model):
        from google.genai import types

        contents, config = self._request(messages, tools)
        model_parts = []
        function_calls = []
        for chunk in self.client.models.generate_content_stream(model=model, contents=contents, config=config):
            for candidate in chunk.candidates or []:
                if not candidate.content:
                    continue
                for part in candidate.content.parts or []:
                    model_parts.append(part)
                    if part.text:
                        yield {"type": "text_delta", "text": part.text}
                    if part.function_call:
                        function_calls.append(part.function_call)

        if function_calls:
            original_content = types.Content(role="model", parts=model_parts)
            for function_call in function_calls:
                yield {
                    "type": "tool_call",
                    "tool_call": self._remember_function_call(original_content, function_call),
                }
        yield {"type": "completed"}


def get_provider(name):
    providers = {"openai": OpenAIProvider, "gemini": GeminiProvider, "groq": GroqProvider, "xai": XAIProvider}
    try:
        return providers[name.lower()]()
    except KeyError as error:
        raise ValueError(f"Unknown provider '{name}'. Choose openai, gemini, groq, or xai.") from error
