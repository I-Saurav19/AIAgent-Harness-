"""Command-line entry point for the coding agent."""

import argparse
import os
import socket
import subprocess
import sys
import time
from urllib.error import URLError
from urllib.request import urlopen
from pathlib import Path

from agent import Agent
from provider import get_provider

GENERATED_WEB_FILES = ["index.html", "style.css", "script.js"]
WEBSITE_TERMS = ("website", "web site", "webpage", "web page", "landing page", "static site")
CREATION_TERMS = ("create", "build", "make", "generate", "design", "develop")


def is_website_creation_request(request):
    text = request.lower()
    return any(term in text for term in WEBSITE_TERMS) and any(term in text for term in CREATION_TERMS)


def reset_generated_web_files():
    for filename in GENERATED_WEB_FILES:
        path = Path(filename)
        if path.is_file() or path.is_symlink():
            path.unlink()
            print(f"🧹 Removed old generated file: {filename}", flush=True)


def _preview_matches(port, page):
    try:
        with urlopen(f"http://127.0.0.1:{port}/", timeout=0.5) as response:
            return response.status == 200 and response.read() == page
    except (OSError, URLError):
        return False


def start_preview_server():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if not all(Path(filename).is_file() for filename in GENERATED_WEB_FILES):
        print("✗ Preview not started: index.html, style.css, and script.js are all required.", flush=True)
        return False
    page_path = Path("index.html")
    page = page_path.read_bytes()

    if _preview_matches(8000, page):
        print("🌐 Live website: http://localhost:8000", flush=True)
        print("✓ Preview server is running", flush=True)
        return True

    try:
        with socket.socket() as probe:
            try:
                probe.bind(("127.0.0.1", 8000))
            except OSError:
                print("✗ Port 8000 is occupied and is not serving this website.", flush=True)
                return False
    except OSError:
        print("✗ Could not reserve a local port for the preview server.", flush=True)
        return False

    options = {
        "cwd": Path.cwd(),
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
    }
    if os.name == "nt":
        options["creationflags"] = (
            getattr(subprocess, "CREATE_NO_WINDOW", 0)
            | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        )
    else:
        options["start_new_session"] = True

    try:
        server = subprocess.Popen(
            [sys.executable, "-m", "http.server", "8000"],
            **options,
        )
    except OSError:
        print("✗ Could not launch the local preview server.", flush=True)
        return False
    for _ in range(30):
        if server.poll() is not None:
            break
        if _preview_matches(8000, page):
            print("🌐 Live website: http://localhost:8000", flush=True)
            print("✓ Preview server is running", flush=True)
            return True
        time.sleep(0.2)

    if server.poll() is None:
        server.terminate()
        server.wait(timeout=2)
    print("✗ Could not start the local preview server.", flush=True)
    return False


def _format_size(byte_count):
    if byte_count < 1024:
        return f"{byte_count} B"
    return f"{byte_count / 1024:.1f} KB"


def _print_limited(label, value, limit=2000):
    if not value:
        return
    value = str(value)
    if len(value) > limit:
        value = value[:limit] + "\n... (output truncated)"
    print(f"{label}:\n{value}", flush=True)


def print_event(event):
    event_type = event["type"]
    if event_type == "agent_start":
        print("🤖 Agent started", flush=True)
    elif event_type == "agent_text":
        if not getattr(print_event, "text_open", False):
            print("\n🤖 Agent: ", end="", flush=True)
            print_event.text_open = True
        print(event.get("text", ""), end="", flush=True)
    elif event_type == "tool_start":
        if getattr(print_event, "text_open", False):
            print(flush=True)
            print_event.text_open = False
        name = event["name"]
        arguments = event.get("arguments", {})
        print(f"\n🔧 Tool: {name}", flush=True)
        if name == "bash":
            print(f"   command: {arguments.get('command', '')}", flush=True)
        elif name == "write_file":
            content = arguments.get("content", "")
            size = len(content.encode("utf-8")) if isinstance(content, str) else 0
            print(f"   path: {arguments.get('path', '')}", flush=True)
            print(f"   size: {_format_size(size)}", flush=True)
        elif name == "read_file":
            path = arguments.get("path", "")
            print(f"   path: {path}", flush=True)
            try:
                print(f"   size: {_format_size(Path(path).stat().st_size)}", flush=True)
            except OSError:
                pass
        elif name == "list_files":
            print(f"   path: {arguments.get('path', '.')}", flush=True)
    elif event_type == "tool_end":
        name = event["name"]
        arguments = event.get("arguments", {})
        result = event.get("result")
        success = event.get("success", False)
        if name == "bash" and isinstance(result, dict):
            status = result.get("exit_code", 1)
            mark = "✓" if status == 0 and success else "✗"
            print(f"{mark} exit_code: {status}", flush=True)
            _print_limited("stdout", result.get("stdout", ""))
            _print_limited("stderr", result.get("stderr", ""))
        elif not success:
            print("✗ Tool failed", flush=True)
            if isinstance(result, dict) and result.get("error"):
                _print_limited("error", result["error"])
        elif name == "write_file":
            print(f"✓ Created {arguments.get('path', 'file')}", flush=True)
        else:
            print("✓ Tool completed", flush=True)
            if name == "list_files" and isinstance(result, list):
                for item in result:
                    print(f"   {item}", flush=True)
            elif name == "read_file" and isinstance(result, str):
                _print_limited("content", result, limit=1000)
            elif isinstance(result, str):
                _print_limited("result", result, limit=1000)
    elif event_type == "agent_end":
        if getattr(print_event, "text_open", False):
            print(flush=True)
            print_event.text_open = False
        print("\n🤖 Agent completed", flush=True)
    elif event_type == "error":
        if getattr(print_event, "text_open", False):
            print(flush=True)
            print_event.text_open = False
        print(f"✗ Agent error: {event.get('error', 'unknown error')}", flush=True)
    elif event_type == "tool_call_error":
        if getattr(print_event, "text_open", False):
            print(flush=True)
            print_event.text_open = False
        print(f"✗ Tool call error ({event.get('name', 'unknown tool')}): {event.get('error', 'arguments could not be parsed')}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Run a small AI coding agent.")
    parser.add_argument("--provider", required=True, choices=["openai", "gemini", "groq", "xai"])
    parser.add_argument("--model", required=True, help="Model name for the selected provider")
    parser.add_argument("request", help="What you want the agent to do")
    args = parser.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print(f"> {args.request}", flush=True)
    if is_website_creation_request(args.request):
        reset_generated_web_files()

    agent = Agent(get_provider(args.provider), args.model)
    agent.run(args.request, on_event=print_event)
    if is_website_creation_request(args.request):
        start_preview_server()


if __name__ == "__main__":
    main()
