#!/usr/bin/env python3
# Harness: context isolation -- protecting the model's clarity of thought.
"""
s04_subagent.py - Subagents

Spawn a child agent with fresh messages=[]. The child works in its own
context, sharing the filesystem, then returns only a summary to the parent.

    Parent agent                     Subagent
    +------------------+             +------------------+
    | messages=[...]   |             | messages=[]      |  <-- fresh
    |                  |  dispatch   |                  |
    | tool: task       | ---------->| while tool_use:  |
    |   prompt="..."   |            |   call tools     |
    |   description="" |            |   append results |
    |                  |  summary   |                  |
    |   result = "..." | <--------- | return last text |
    +------------------+             +------------------+
              |
    Parent context stays clean.
    Subagent context is discarded.

Key insight: "Process isolation gives context isolation for free."
"""

import os
import subprocess
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv(override=True)

WORKDIR = Path.cwd()
client = genai.Client()
MODEL = os.environ["MODEL_ID"]

SYSTEM = f"You are a coding agent at {WORKDIR}. Use the task tool to delegate exploration or subtasks."
SUBAGENT_SYSTEM = f"You are a coding subagent at {WORKDIR}. Complete the given task, then summarize your findings."


# -- Tool implementations shared by parent and child --
def safe_path(p: str) -> Path:
    path = (WORKDIR / p).resolve()
    if not path.is_relative_to(WORKDIR):
        raise ValueError(f"Path escapes workspace: {p}")
    return path

def run_bash(command: str) -> str:
    dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]
    if any(d in command for d in dangerous):
        return "Error: Dangerous command blocked"
    try:
        r = subprocess.run(command, shell=True, cwd=WORKDIR,
                           capture_output=True, text=True, timeout=120)
        out = (r.stdout + r.stderr).strip()
        return out[:50000] if out else "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: Timeout (120s)"

def run_read(path: str, limit: int = None) -> str:
    try:
        lines = safe_path(path).read_text().splitlines()
        if limit and limit < len(lines):
            lines = lines[:limit] + [f"... ({len(lines) - limit} more)"]
        return "\n".join(lines)[:50000]
    except Exception as e:
        return f"Error: {e}"

def run_write(path: str, content: str) -> str:
    try:
        fp = safe_path(path)
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(content)
        return f"Wrote {len(content)} bytes"
    except Exception as e:
        return f"Error: {e}"

def run_edit(path: str, old_text: str, new_text: str) -> str:
    try:
        fp = safe_path(path)
        content = fp.read_text()
        if old_text not in content:
            return f"Error: Text not found in {path}"
        fp.write_text(content.replace(old_text, new_text, 1))
        return f"Edited {path}"
    except Exception as e:
        return f"Error: {e}"


TOOL_HANDLERS = {
    "bash":       lambda **kw: run_bash(kw["command"]),
    "read_file":  lambda **kw: run_read(kw["path"], kw.get("limit")),
    "write_file": lambda **kw: run_write(kw["path"], kw["content"]),
    "edit_file":  lambda **kw: run_edit(kw["path"], kw["old_text"], kw["new_text"]),
}

# Child gets all base tools except task (no recursive spawning)
CHILD_TOOL_DECLARATIONS = [
    {"name": "bash", "description": "Run a shell command.",
     "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}},
    {"name": "read_file", "description": "Read file contents.",
     "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "limit": {"type": "integer"}}, "required": ["path"]}},
    {"name": "write_file", "description": "Write content to file.",
     "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}},
    {"name": "edit_file", "description": "Replace exact text in file.",
     "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "old_text": {"type": "string"}, "new_text": {"type": "string"}}, "required": ["path", "old_text", "new_text"]}},
]


# -- Subagent: fresh context, filtered tools, summary-only return --
def run_subagent(prompt: str) -> str:
    sub_messages = [{"role": "user", "parts": [{"text": prompt}]}]  # fresh context
    last_parts = []
    for _ in range(30):  # safety limit
        config = types.GenerateContentConfig(
            system_instruction=SUBAGENT_SYSTEM,
            tools=[types.Tool(function_declarations=CHILD_TOOL_DECLARATIONS)]
        )
        response = client.models.generate_content(
            model=MODEL, contents=sub_messages, config=config,
        )
        last_parts = response.candidates[0].content.parts
        sub_messages.append({"role": "model", "parts": last_parts})
        function_calls = [p for p in last_parts if p.function_call is not None]
        if not function_calls:
            break
        result_parts = []
        for p in function_calls:
            fc = p.function_call
            handler = TOOL_HANDLERS.get(fc.name)
            print(f"[subagent] {fc.name} 工具调用，入参：{fc.args}")
            output = handler(**fc.args) if handler else f"Unknown tool: {fc.name}"
            print(f"  输出：{output}")
            result_parts.append(types.Part(
                function_response=types.FunctionResponse(
                    name=fc.name,
                    response={"result": str(output)[:50000]}
                )
            ))
        sub_messages.append({"role": "user", "parts": result_parts})
    # Only the final text returns to the parent -- child context is discarded
    summary = "".join(p.text for p in last_parts if hasattr(p, "text") and p.text)
    return summary or "(no summary)"


# -- Parent tools: base tools + task dispatcher --
PARENT_TOOL_DECLARATIONS = CHILD_TOOL_DECLARATIONS + [
    {"name": "task", "description": "Spawn a subagent with fresh context. It shares the filesystem but not conversation history.",
     "parameters": {"type": "object", "properties": {"prompt": {"type": "string"}, "description": {"type": "string", "description": "Short description of the task"}}, "required": ["prompt"]}},
]


def agent_loop(messages: list):
    while True:
        config = types.GenerateContentConfig(
            system_instruction=SYSTEM,
            tools=[types.Tool(function_declarations=PARENT_TOOL_DECLARATIONS)]
        )
        response = client.models.generate_content(
            model=MODEL, contents=messages, config=config,
        )
        parts = response.candidates[0].content.parts
        messages.append({"role": "model", "parts": parts})
        function_calls = [p for p in parts if p.function_call is not None]
        if not function_calls:
            return
        result_parts = []
        for p in function_calls:
            fc = p.function_call
            print(f"[main] {fc.name} 工具调用，入参：{fc.args}")
            if fc.name == "task":
                desc = fc.args.get("description", "subtask")
                print(f"> task ({desc}): {fc.args['prompt'][:80]}")
                output = run_subagent(fc.args["prompt"])
            else:
                handler = TOOL_HANDLERS.get(fc.name)
                output = handler(**fc.args) if handler else f"Unknown tool: {fc.name}"
            print(f"  输出  {str(output)[:200]}")
            result_parts.append(types.Part(
                function_response=types.FunctionResponse(
                    name=fc.name,
                    response={"result": str(output)}
                )
            ))
        messages.append({"role": "user", "parts": result_parts})


if __name__ == "__main__":
    history = []
    while True:
        try:
            query = input("\033[36ms04 >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break
        if query.strip().lower() in ("q", "exit", ""):
            break
        history.append({"role": "user", "parts": [{"text": query}]})
        agent_loop(history)
        for part in history[-1]["parts"]:
            if hasattr(part, "text") and part.text:
                print(part.text)
        print()
