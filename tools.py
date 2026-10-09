"""Small workspace tools exposed to the model."""

import subprocess
from pathlib import Path
import json


def read_file(path):
    return Path(path).read_text(encoding="utf-8")


def write_file(path, content):
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(content, encoding="utf-8")
    return f"Wrote {file_path}"


def bash(command):
    result = subprocess.run(command, shell=True, capture_output=True, text=True, cwd=Path.cwd())
    return {"stdout": result.stdout, "stderr": result.stderr, "exit_code": result.returncode}


def list_files(path="."):
    return [item.name + ("/" if item.is_dir() else "") for item in sorted(Path(path).iterdir())]


TOOLS = {"read_file": read_file, "write_file": write_file, "bash": bash, "list_files": list_files}

TOOL_DEFINITIONS = [
    {
        "name": "read_file",
        "description": "Read and return a UTF-8 text file from the workspace.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path of the file to read, relative to the current directory when not absolute."}
            },
            "required": ["path"],
        },
    },
    {
        "name": "write_file",
        "description": "Create or overwrite a file in the workspace with the supplied complete content.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path of the file to create or overwrite."},
                "content": {"type": "string", "description": "Complete content to write into the file."},
            },
            "required": ["path", "content"],
        },
    },
    {
        "name": "bash",
        "description": "Run a shell command in the current working directory and return stdout, stderr, and exit code.",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "Shell command to execute."}
            },
            "required": ["command"],
        },
    },
    {
        "name": "list_files",
        "description": "List the files and directories at a workspace path.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Directory path to list; defaults to the current directory."}
            },
        },
    },
]


def execute_tool(name, arguments):
    function = TOOLS.get(name)
    if function is None:
        return {"error": f"Unknown tool: {name}"}
    if not isinstance(arguments, dict):
        return {"error": f"Arguments for {name} must be an object."}

    definition = next((item for item in TOOL_DEFINITIONS if item["name"] == name), {})
    allowed = definition.get("parameters", {}).get("properties", {})
    arguments = {key: value for key, value in arguments.items() if key in allowed}
    try:
        return function(**arguments)
    except Exception as error:
        return {"error": str(error)}


def result_as_text(result):
    return result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)
