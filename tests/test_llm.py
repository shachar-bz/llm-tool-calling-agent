from types import SimpleNamespace

import pytest
from fakes import ScriptedTransport, final_reply

from tool_agent.llm import LLM, CallCapReached, OpenAITransport


def test_llm_refuses_requests_past_the_cap():
    transport = ScriptedTransport(final_reply("one"), final_reply("two"), final_reply("three"))
    llm = LLM(transport, max_calls=2)

    assert llm.complete([]).content == "one"
    assert llm.complete([]).content == "two"
    with pytest.raises(CallCapReached):
        llm.complete([])
    assert len(transport.requests) == 2


def test_openai_transport_sends_the_model_and_returns_the_message():
    sent = {}

    def create(**request):
        sent.update(request)
        return SimpleNamespace(choices=[SimpleNamespace(message=final_reply("hi"))])

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    transport = OpenAITransport(client, "gpt-test")

    reply = transport(messages=[{"role": "user", "content": "hello"}], tool_choice="auto")

    assert reply.content == "hi"
    assert sent == {
        "model": "gpt-test",
        "messages": [{"role": "user", "content": "hello"}],
        "tool_choice": "auto",
    }
