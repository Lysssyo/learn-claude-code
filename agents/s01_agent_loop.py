#!/usr/bin/env python3
# Harness: the loop -- the model's first connection to the real world.
"""
s01_agent_loop.py - The Agent Loop

The entire secret of an AI coding agent in one pattern:

    while stop_reason == "tool_use":
        response = LLM(messages, tools)
        execute tools
        append results

    +----------+      +-------+      +---------+
    |   User   | ---> |  LLM  | ---> |  Tool   |
    |  prompt  |      |       |      | execute |
    +----------+      +---+---+      +----+----+
                          ^               |
                          |   tool_result |
                          +---------------+
                          (loop continues)

This is the core loop: feed tool results back to the model
until the model decides to stop. Production agents layer
policy, hooks, and lifecycle controls on top.
"""

import os
import subprocess

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv(override=True)

client = genai.Client()
MODEL = os.environ["MODEL_ID"]

SYSTEM = f"You are a coding agent at {os.getcwd()}. Use bash to solve tasks. Act, don't explain."

bash_func = {
    "name": "bash",
    "description": "Run a shell command.",
    "parameters": {
        "type": "object",
        "properties": {"command": {"type": "string"}},
        "required": ["command"],
    },
}


def run_bash(command: str) -> str:
    dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]
    if any(d in command for d in dangerous):
        return "Error: Dangerous command blocked"
    try:
        r = subprocess.run(command, shell=True, cwd=os.getcwd(),
                           capture_output=True, text=True, timeout=120)
        out = (r.stdout + r.stderr).strip()
        return out[:50000] if out else "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: Timeout (120s)"


count = 0


# -- The core pattern: a while loop that calls tools until the model stops --
def agent_loop(messages: list):
    global count
    while True:
        count += 1
        print(f"进入{count}次loop")

        config = types.GenerateContentConfig(
            system_instruction=SYSTEM,
            tools=[types.Tool(function_declarations=[bash_func])]
        )
        response = client.models.generate_content(
            model=MODEL,
            contents=messages,
            config=config,
        )
        # Append assistant turn (Google uses "model" role, not "assistant")
        parts = response.candidates[0].content.parts
        messages.append({"role": "model", "parts": parts})

        # If the model didn't call a tool, we're done
        function_calls = [p for p in parts if p.function_call]
        if not function_calls:
            return

        # Execute each tool call, collect results
        result_parts = []
        for part in function_calls:
            fc = part.function_call
            print(f"\033[33m$ {fc.args['command']}\033[0m")
            output = run_bash(fc.args["command"])
            print(output[:200])
            result_parts.append(types.Part(
                function_response=types.FunctionResponse(
                    name=fc.name,
                    response={"result": output}
                )
            ))
        messages.append({"role": "user", "parts": result_parts})


if __name__ == "__main__":

    history = []
    while True:
        try:
            query = input("\033[36ms01 >> \033[0m")
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
