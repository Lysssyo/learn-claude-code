#!/usr/bin/env python3
# Harness: compression -- clean memory for infinite sessions.
"""
s06_context_compact.py - Compact

Three-layer compression pipeline so the agent can work forever:

    Every turn:
    +------------------+
    | Tool call result |
    +------------------+
            |
            v
    [Layer 1: micro_compact]        (silent, every turn)
      Replace tool_result content older than last 3
      with "[Previous: used {tool_name}]"
            |
            v
    [Check: tokens > 50000?]
       |               |
       no              yes
       |               |
       v               v
    continue    [Layer 2: auto_compact]
                  Save full transcript to .transcripts/
                  Ask LLM to summarize conversation.
                  Replace all messages with [summary].
                        |
                        v
                [Layer 3: compact tool]
                  Model calls compact -> immediate summarization.
                  Same as auto, triggered manually.

Key insight: "The agent can forget strategically and keep working forever."
"""

import json
import os
import subprocess
import time
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv(override=True)

WORKDIR = Path.cwd()
client = genai.Client()
MODEL = os.environ["MODEL_ID"]

print(WORKDIR)

SYSTEM = f"You are a coding agent at {WORKDIR}. Use tools to solve tasks."

THRESHOLD = 50000
TRANSCRIPT_DIR = WORKDIR / ".transcripts"
KEEP_RECENT = 3


def estimate_tokens(messages: list) -> int:
    """Rough token count: ~4 chars per token."""
    return len(str(messages)) // 4


# -- Layer 1: micro_compact - replace old tool results with placeholders --
def micro_compact(messages: list) -> list:
    # Collect (msg_index, part_index, Part) for all function_response parts in "tool" role messages
    tool_results = []
    for msg_idx, msg in enumerate(messages):
        if msg.get("role") == "user":
            parts = msg.get("parts", [])
            for part_idx, part in enumerate(parts):
                # part may be a types.Part object or a dict
                if hasattr(part, "function_response") and part.function_response is not None:
                    tool_results.append((msg_idx, part_idx, part))
    if len(tool_results) <= KEEP_RECENT:
        return messages
    # Clear old results (keep last KEEP_RECENT)
    to_clear = tool_results[:-KEEP_RECENT]
    for msg_idx, part_idx, part in to_clear:
        fr = part.function_response
        tool_name = fr.name if fr else "unknown"
        result_str = str(fr.response.get("result", "")) if fr and fr.response else ""
        if len(result_str) > 100:
            # Replace with a placeholder Part
            messages[msg_idx]["parts"][part_idx] = types.Part(
                function_response=types.FunctionResponse(
                    name=tool_name,
                    response={"result": f"[Previous: used {tool_name}]"}
                )
            )
    return messages


# -- Layer 2: auto_compact - save transcript, summarize, replace messages --
def auto_compact(messages: list) -> list:
    # Save full transcript to disk
    TRANSCRIPT_DIR.mkdir(exist_ok=True)
    transcript_path = TRANSCRIPT_DIR / f"transcript_{int(time.time())}.jsonl"
    with open(transcript_path, "w") as f:
        for msg in messages:
            f.write(json.dumps(msg, default=str) + "\n")
    print(f"[transcript saved: {transcript_path}]")
    # Ask LLM to summarize
    conversation_text = json.dumps(messages, default=str)[:80000]
    summary_messages = [{"role": "user", "parts": [{"text":
                                                        "Summarize this conversation for continuity. Include: "
                                                        "1) What was accomplished, 2) Current state, 3) Key decisions made. "
                                                        "Be concise but preserve critical details.\n\n" + conversation_text
                                                    }]}]
    response = client.models.generate_content(
        model=MODEL,
        contents=summary_messages,
    )

    # summary_parts = response.candidates[0].content.parts
    # summary = "".join(p.text for p in summary_parts if hasattr(p, "text") and p.text)
    summary = response.parts
    # Replace all messages with compressed summary
    return [
        {"role": "user", "parts": [{"text": f"[Conversation compressed. Transcript: {transcript_path}]\n\n{summary}"}]},
        {"role": "model", "parts": [types.Part(text="Understood. I have the context from the summary. Continuing.")]},
    ]


# -- Tool implementations --
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
    "bash": lambda **kw: run_bash(kw["command"]),
    "read_file": lambda **kw: run_read(kw["path"], kw.get("limit")),
    "write_file": lambda **kw: run_write(kw["path"], kw["content"]),
    "edit_file": lambda **kw: run_edit(kw["path"], kw["old_text"], kw["new_text"]),
    "compact": lambda **kw: "Manual compression requested.",
}

TOOL_DECLARATIONS = [
    {"name": "bash", "description": "Run a shell command.",
     "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}},
    {"name": "read_file", "description": "Read file contents.",
     "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "limit": {"type": "integer"}},
                    "required": ["path"]}},
    {"name": "write_file", "description": "Write content to file.",
     "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                    "required": ["path", "content"]}},
    {"name": "edit_file", "description": "Replace exact text in file.",
     "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "old_text": {"type": "string"},
                                                     "new_text": {"type": "string"}},
                    "required": ["path", "old_text", "new_text"]}},
    {"name": "compact", "description": "Trigger manual conversation compression.",
     "parameters": {"type": "object",
                    "properties": {"focus": {"type": "string", "description": "What to preserve in the summary"}}}},
]


def agent_loop(messages: list):
    while True:
        # Layer 1: micro_compact before each LLM call
        micro_compact(messages)
        # Layer 2: auto_compact if token estimate exceeds threshold
        if estimate_tokens(messages) > THRESHOLD:
            print("[auto_compact triggered]")
            messages[:] = auto_compact(messages)
        config = types.GenerateContentConfig(
            system_instruction=SYSTEM,
            tools=[types.Tool(function_declarations=TOOL_DECLARATIONS)]
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
        manual_compact = False
        for p in function_calls:
            fc = p.function_call
            if fc.name == "compact":
                manual_compact = True
                output = "Compressing..."
            else:
                handler = TOOL_HANDLERS.get(fc.name)
                try:
                    output = handler(**fc.args) if handler else f"Unknown tool: {fc.name}"
                except Exception as e:
                    output = f"Error: {e}"
            print(f"> {fc.name}: {str(output)[:200]}")
            result_parts.append(types.Part(
                function_response=types.FunctionResponse(
                    name=fc.name,
                    response={"result": str(output)}
                )
            ))
        messages.append({"role": "user", "parts": result_parts})
        # Layer 3: manual compact triggered by the compact tool
        if manual_compact:
            print("[manual compact]")
            messages[:] = auto_compact(messages)


if __name__ == "__main__":
    history = []
    while True:
        try:
            query = input("\033[36ms06 >> \033[0m")
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
