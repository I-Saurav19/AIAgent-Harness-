"""Provider-independent coding agent loop."""

import json

from tools import TOOL_DEFINITIONS, execute_tool, result_as_text

SYSTEM_PROMPT = """You are an autonomous coding agent.

The user's workspace is the current working directory. You can use tools to inspect and modify it.

When the user asks you to create, modify, or delete files, you MUST use the available tools to perform the operation. Do not merely provide code in your response. Apply this rule to all coding tasks.

For a static website task, create separate HTML, CSS, and JavaScript files when appropriate, using index.html, style.css, and script.js. Do not put all CSS and JavaScript into index.html unless the user specifically requests a single-file website. Use write_file to actually create the files. Do not merely output the code in your final response.

Use read_file and list_files to inspect the workspace when needed. Use bash to run commands when needed. After using a tool, examine its result. If it reports an error or does not confirm the requested operation, take the next appropriate action instead of claiming success.

Only provide a localhost preview URL when a bash result confirms that the server started successfully. Never claim that a server is running when the command failed or did not confirm a successful start.

Never claim that you changed a file unless you actually used a tool to change it. Never claim that a command was executed unless the bash tool actually executed it. If a response includes tool calls, make those calls even when it also includes text, then continue the conversation. Only finish when there are no more tool calls and the task is complete. When complete, provide a short summary."""


class Agent:
    def __init__(self, provider, model):
        self.provider = provider
        self.model = model

    def run(self, user_request, on_event=None):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_request},
        ]
        self._emit(on_event, "agent_start")
        tool_call_retries = 0

        while True:
            text_parts = []
            tool_calls = []
            tool_call_errors = []
            try:
                for event in self.provider.stream(messages=messages, tools=TOOL_DEFINITIONS, model=self.model):
                    if event["type"] == "text_delta":
                        text_parts.append(event.get("text", ""))
                        self._emit(on_event, "agent_text", text=event.get("text", ""))
                    elif event["type"] == "tool_call":
                        tool_calls.append(event["tool_call"])
                    elif event["type"] == "tool_call_error":
                        error = {
                            "name": event.get("name", "unknown tool"),
                            "error": event.get("error", "The tool call could not be parsed."),
                        }
                        tool_call_errors.append(error)
                        self._emit(on_event, "tool_call_error", **error)
            except Exception as error:
                self._emit(on_event, "error", error=str(error))
                raise

            text = "".join(text_parts)
            if not tool_calls:
                if tool_call_errors:
                    if tool_call_retries >= 1:
                        final_text = (
                            "I couldn't complete the request because Groq returned malformed tool-call "
                            "arguments again. No tool was executed for the invalid call."
                        )
                        self._emit(on_event, "agent_text", text=final_text)
                        self._emit(on_event, "agent_end", text=final_text)
                        return final_text
                    if text:
                        messages.append({"role": "assistant", "content": text})
                    problem = "; ".join(f"{item['name']}: {item['error']}" for item in tool_call_errors)
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                f"Your tool call could not be used: {problem} "
                                "Retry with complete, valid JSON arguments matching the tool schema. "
                                "Do not include partial JSON."
                            ),
                        }
                    )
                    tool_call_retries += 1
                    continue
                self._emit(on_event, "agent_end", text=text)
                return text

            messages.append(
                {
                    "role": "assistant",
                    "content": text or None,
                    "tool_calls": [
                        {
                            "id": call["id"],
                            "type": "function",
                            "function": {
                                "name": call["name"],
                                "arguments": json.dumps(call["arguments"]),
                            },
                        }
                        for call in tool_calls
                    ],
                }
            )

            for call in tool_calls:
                arguments = call.get("arguments") or {}
                self._emit(on_event, "tool_start", name=call["name"], arguments=arguments)
                result = execute_tool(call["name"], arguments)
                success = not (isinstance(result, dict) and (
                    "error" in result or result.get("exit_code", 0) != 0
                ))
                self._emit(
                    on_event,
                    "tool_end",
                    name=call["name"],
                    arguments=arguments,
                    result=result,
                    success=success,
                )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call["id"],
                        "name": call["name"],
                        "content": result_as_text(result),
                    }
                )

            if tool_call_errors:
                if tool_call_retries >= 1:
                    final_text = (
                        "I couldn't complete the request because Groq returned malformed tool-call "
                        "arguments again. No tool was executed for the invalid call."
                    )
                    self._emit(on_event, "agent_text", text=final_text)
                    self._emit(on_event, "agent_end", text=final_text)
                    return final_text
                tool_call_retries += 1
                problem = "; ".join(f"{item['name']}: {item['error']}" for item in tool_call_errors)
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"Your tool call could not be used: {problem} "
                            "Retry with complete, valid JSON arguments matching the tool schema."
                        ),
                    }
                )
            else:
                tool_call_retries = 0

    @staticmethod
    def _emit(callback, event_type, **data):
        if callback:
            callback({"type": event_type, **data})
