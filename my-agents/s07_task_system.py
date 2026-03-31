#!/usr/bin/env python3
import json
import os
import subprocess
import shutil
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv(override=True)

WORKDIR = Path.cwd()
TASK_DIR = WORKDIR / "tasks"

client = genai.Client()
MODEL = os.environ["MODEL_ID"]

SYSTEM = f"You are a coding agent at {WORKDIR}. Use tools to solve tasks."


class TaskManager:
    def __init__(self, task_dir: Path):
        self.task_dir = task_dir
        self._init_dir()
        self.next_id = self._get_max_task_id() + 1

    def _init_dir(self):
        self.task_dir.mkdir(exist_ok=True)

    def _get_max_task_id(self):
        max_id = 0
        for f in Path(self.task_dir).glob("task_*.json"):
            task_id = int(f.stem.split("_")[1])
            max_id = task_id if task_id > max_id else max_id
        return max_id

    def _load(self, task_id: int):
        """从磁盘读取一个任务json"""
        f_name = f"task_{task_id}.json"
        with open(self.task_dir / f_name, "r") as f:
            data = json.load(f)
        return data

    def _save(self, task):
        f_name = f"task_{task['id']}.json"
        with open(self.task_dir / f_name, "w") as f:
            json.dump(task, f, indent=2)

    def _clear_dependency(self, completed_id):
        for task_id in range(1, self.next_id):
            task = self._load(task_id)
            if completed_id in task["blockedBy"]:
                task["blockedBy"].remove(completed_id)
            self._save(task)

    def create(self, subject: str, description=""):
        task_detail = {"id": self.next_id, "subject": subject, "description": description, "status": "pending",
                       "blockedBy": [],
                       "blocks": []}
        self._save(task_detail)
        self.next_id += 1
        return f"Created task #{task_detail['id']}: {subject}"

    def get(self, task_id: int) -> dict:
        return self._load(task_id)

    def list_all(self):
        if self.next_id <= 1:
            return "No tasks."
        lines = []
        for task_id in range(1, self.next_id):
            task_detail = self.get(task_id)
            marker = {"pending": "[ ]", "in_progress": "[>]", "completed": "[x]"}.get(task_detail["status"], "[?]")
            blocked = f" (blocked by: {task_detail['blockedBy']})" if task_detail.get("blockedBy") else ""
            lines.append(f"{marker} #{task_detail['id']}: {task_detail['subject']}{blocked}")
        return "\n".join(lines)

    def update(self, task_id, status=None, add_blocked_by=None, add_blocks=None):
        task_detail = self.get(task_id)
        if status:
            task_detail["status"] = status
            if status == "completed":
                self._clear_dependency(task_id)
        if add_blocked_by:
            task_detail["blockedBy"] = list(set(task_detail["blockedBy"] + add_blocked_by))
        if add_blocks:
            task_detail["blocks"].append(add_blocks)
            for be_blocked in add_blocks:
                task = self._load(be_blocked)
                task["blockedBy"] = list(set(task["blockedBy"] + [task_id]))
                self._save(task)  # 写回磁盘

        self._save(task_detail)
        return json.dumps(task_detail, indent=2)


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
        text = safe_path(path).read_text()
        lines = text.splitlines()
        if limit and limit < len(lines):
            lines = lines[:limit] + [f"... ({len(lines) - limit} more lines)"]
        return "\n".join(lines)[:50000]
    except Exception as e:
        return f"Error: {e}"


def run_write(path: str, content: str) -> str:
    try:
        fp = safe_path(path)
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(content)
        return f"Wrote {len(content)} bytes to {path}"
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


TASK_MANAGER = TaskManager(TASK_DIR)

# -- The dispatch map: {tool_name: handler} --
TOOL_HANDLERS = {
    "bash": lambda **kw: run_bash(kw["command"]),
    "read_file": lambda **kw: run_read(kw["path"], kw.get("limit")),
    "write_file": lambda **kw: run_write(kw["path"], kw["content"]),
    "edit_file": lambda **kw: run_edit(kw["path"], kw["old_text"], kw["new_text"]),
    "task_create": lambda **kw: TASK_MANAGER.create(kw["subject"], kw.get("description", "")),
    "task_update": lambda **kw: TASK_MANAGER.update(kw["task_id"], kw.get("status"), kw.get("addBlockedBy"),
                                                    kw.get("addBlocks")),
    "task_list": lambda **kw: TASK_MANAGER.list_all(),
    "task_get": lambda **kw: TASK_MANAGER.get(kw["task_id"]),
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
    {"name": "task_create", "description": "Create a new task.",
     "parameters": {"type": "object", "properties": {"subject": {"type": "string"}, "description": {"type": "string"}},
                    "required": ["subject"]}},
    {"name": "task_update", "description": "Update a task's status or dependencies.",
     "parameters": {"type": "object", "properties": {
         "task_id": {"type": "integer"},
         "status": {"type": "string", "enum": ["pending", "in_progress", "completed"]},
         "addBlockedBy": {"type": "array", "items": {"type": "integer"}},
         "addBlocks": {"type": "array", "items": {"type": "integer"}}},
                    "required": ["task_id"]}},
    {"name": "task_list", "description": "List all tasks with status summary.",
     "parameters": {"type": "object", "properties": {}}},
    {"name": "task_get", "description": "Get full details of a task by ID.",
     "parameters": {"type": "object", "properties": {"task_id": {"type": "integer"}},
                    "required": ["task_id"]}},
]


def agent_loop(messages: list):
    while True:
        config = types.GenerateContentConfig(
            system_instruction=SYSTEM,
            tools=[types.Tool(function_declarations=TOOL_DECLARATIONS)]
        )
        response = client.models.generate_content(
            model=MODEL, contents=messages, config=config,
        )
        parts = response.candidates[0].content.parts
        messages.append({"role": "model", "parts": parts})

        function_calls = [p for p in parts if p.function_call]
        if not function_calls:
            return

        result_parts = []
        for p in function_calls:
            fc = p.function_call

            handler = TOOL_HANDLERS.get(fc.name)
            try:
                output = handler(**fc.args) if handler else f"Unknown tool: {fc.name}"
            except Exception as e:
                output = f"Error: {e}"
            output = str(output) if output else "(no output)"
            print(f"> {fc.name}: {output[:200]}")
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
            query = input("\033[36ms07 >> \033[0m")
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
