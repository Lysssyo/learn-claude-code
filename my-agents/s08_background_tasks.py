#!/usr/bin/env python3
import json
import os
import subprocess
import threading
import uuid
from pathlib import Path
import time

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv(override=True)

WORKDIR = Path.cwd()
client = genai.Client()
MODEL = os.environ["MODEL_ID"]

SYSTEM = f"You are a coding agent at {WORKDIR}. Use tools to solve tasks. Act, don't explain."


class BackgroundManager:
    def __init__(self):
        self.tasks = {}
        self._notification_queue = []
        self._lock = threading.Lock()

    def run(self, command):
        task_id = str(uuid.uuid4())[:8]
        self.tasks[task_id] = {"status": "running", "result": None, "command": command}
        threading.Thread(target=self._execute, args=(task_id, command), daemon=True).start()
        # return f"Background task {task_id} started:{command}"
        return task_id

    def _execute(self, task_id, command):
        dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]

        if any(d in command for d in dangerous):
            result = "dangerous command can not be executed"
            status = "error"
        else:
            try:
                r = subprocess.run(command, shell=True, cwd=WORKDIR,
                                   capture_output=True, text=True, timeout=300)
                out = (r.stdout + r.stderr).strip()
                result = out[:50000] if out else "(no output)"
                status = "completed"
            except subprocess.TimeoutExpired:
                result = "Error: Timeout (120s)"
                status = "timeout"

        self.tasks[task_id]["status"] = status
        self.tasks[task_id]["result"] = result

        with self._lock:
            self._notification_queue.append({
                "task_id": task_id,
                "status": status,
                "command": command[:80],
                "result": (result or "(no output)")[:500],
            })

    def _list_all_background_tasks(self) -> list[dict]:
        """ 返回值：
         [
             {"task_id": "abc12345", "status": "completed", "command": "sleep 5 && echo done", "result": "done"},
             {"task_id": "def67890", "status": "completed", "command": "make build", "result": "Build successful"}
        ]
        """
        task_list = []
        for key, value in self.tasks.items():
            task_detail = {"task_id": key, "status": value["status"], "command": value["command"],
                           "result": value["result"]}
            task_list.append(task_detail)
        return task_list

    def check(self, task_id=None) -> str:
        """LLM工具，返回人类可读字符串：abc12345: [completed] sleep 5 && echo done"""
        if task_id is None:
            task_list = self._list_all_background_tasks()
            if not task_list:
                return "Error No background tasks."
            # 转为人类可读字符串
            tasks_str = ""
            for item in task_list:
                tasks_str += f"{item['task_id']} : [{item['status']}] {item['command']}  {item['result']} \n"
            return tasks_str
        else:
            task = self.tasks.get(task_id)
            if not task:
                return f"Error: Unknown task {task_id}"
            task_str = f"{task_id} : [{task['status']}] {task['command']}  {task['result']} \n"
            return task_str

    def drain_notifications(self) -> list[dict]:
        """非LLM工具，返回结构化数据"""
        with self._lock:
            notifs = list(self._notification_queue)
            self._notification_queue.clear()
        return notifs


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

    # -- The dispatch map: {tool_name: handler} --


TOOL_HANDLERS = {
    "bash": lambda **kw: run_bash(kw["command"]),
    "read_file": lambda **kw: run_read(kw["path"], kw.get("limit")),
    "write_file": lambda **kw: run_write(kw["path"], kw["content"]),
    "edit_file": lambda **kw: run_edit(kw["path"], kw["old_text"], kw["new_text"]),
    "background_run": lambda **kw: BM.run(kw["command"]),
    "check_background": lambda **kw: BM.check(kw.get("task_id")),
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
    {"name": "background_run", "description": "Run command in background thread (non-blocking). Returns task_id immediately.",
     "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}},
    {"name": "check_background", "description": "Check background task status. Omit task_id to list all.",
     "parameters": {"type": "object", "properties": {"task_id": {"type": "string"}}}},
]

BM = BackgroundManager()


def agent_loop(messages: list):
    while True:

        notifs = BM.drain_notifications()

        if notifs:
            notifs_str = "<background-results>"

            for item in notifs:
                # 每一个都是字典
                notifs_str += f"{item['task_id']} : [{item['status']}] {item['command']}  {item['result']} \n"

            notifs_str += "</background-results>"

            msg1 = {"role": "user", "parts": [types.Part(
                text=notifs_str
            )]}

            msg2 = {"role": "model", "parts": [types.Part(
                text="Noted background results."
            )]}

            messages.append(msg1)
            messages.append(msg2)

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
            output = handler(**fc.args) if handler else f"Unknown tool: {fc.name}"
            print(f"> {fc.name}: {output[:200]}")
            result_parts.append(types.Part(
                function_response=types.FunctionResponse(
                    name=fc.name,
                    response={"result": output}
                )
            ))
        messages.append({"role": "user", "parts": result_parts})


if __name__ == "__main__":
    # 测试 1：启动后台任务
    # task_id = BM.run("sleep 2 && echo hello")
    # assert task_id is not None
    # assert len(task_id) == 8
    # print(f"✓ 测试1：启动后台任务，task_id={task_id}")
    #
    # # 测试 2：刚启动时状态是 running
    # status = BM.check(task_id)
    # assert "running" in status
    # print(f"✓ 测试2：状态为 running")
    #
    # # 测试 3：通知队列暂时为空（任务还没完成）
    # notifs = BM.drain_notifications()
    # assert notifs == []
    # print(f"✓ 测试3：通知队列暂时为空")
    #
    # # 测试 4：等待任务完成
    # time.sleep(3)
    # status = BM.check(task_id)
    # assert "completed" in status
    # assert "hello" in status
    # print(f"✓ 测试4：任务完成，结果包含 hello")
    #
    # # 测试 5：通知队列有数据
    # notifs = BM.drain_notifications()
    # assert len(notifs) == 1
    # assert notifs[0]["task_id"] == task_id
    # print(f"✓ 测试5：通知队列有 1 条通知")
    #
    # # 测试 6：drain 后队列清空
    # notifs = BM.drain_notifications()
    # assert notifs == []
    # print(f"✓ 测试6：drain 后队列清空")
    #
    # # 测试 7：多任务并行
    # id1 = BM.run("echo aaa")
    # id2 = BM.run("echo bbb")
    # time.sleep(2)
    # all_status = BM.check()
    # assert id1 in all_status
    # assert id2 in all_status
    # print(f"✓ 测试7：多任务并行")
    #
    # # 测试 8：check 不存在的 task_id
    # result = BM.check("nonexist")
    # assert "Error" in result or "Unknown" in result
    # print(f"✓ 测试8：不存在的 task_id")
    #
    # print("\n全部测试通过！")

    history = []
    while True:
        try:
            query = input("\033[36ms08 >> \033[0m")
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
