import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.memory import ConversationMemory

def test_add_and_length():
    mem = ConversationMemory()
    mem.add_user("Hi")
    mem.add_assistant("Hello!")
    assert len(mem) == 2

def test_system_prompt_in_messages():
    mem = ConversationMemory(system_prompt="Be concise.")
    mem.add_user("Hi")
    msgs = mem.to_messages()
    assert msgs[0] == {"role": "system", "content": "Be concise."}
    assert msgs[1]["role"] == "user"

def test_anthropic_format():
    mem = ConversationMemory(system_prompt="Sys")
    mem.add_user("Hi")
    sys_p, msgs = mem.to_anthropic_messages()
    assert sys_p == "Sys"
    assert all(m["role"] in ("user", "assistant") for m in msgs)

def test_eviction():
    mem = ConversationMemory(max_turns=2)
    for i in range(5):
        mem.add_user(f"msg {i}")
        mem.add_assistant(f"reply {i}")
    assert len(mem) == 4
    assert mem._history[-2].content == "msg 4"

def test_clear():
    mem = ConversationMemory(system_prompt="Sys")
    mem.add_user("Hi")
    mem.clear()
    assert len(mem) == 0
    assert mem.system_prompt == "Sys"