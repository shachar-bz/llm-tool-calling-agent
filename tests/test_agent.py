import io
import json

from support import ScriptedTransport, final_reply, tool_call_reply

from tool_agent.agent import Agent
from tool_agent.llm import LLM, LLMRejected
from tool_agent.run_log import RunLog
from tool_agent.tools import Tool, ToolRegistry
from tool_agent.tools.calculator import CALCULATOR


def make_agent(replies, tools=(CALCULATOR,), max_llm_calls=20, max_tool_calls=20):
    transport = ScriptedTransport(*replies)
    llm = LLM(transport, max_llm_calls)
    output = io.StringIO()
    agent = Agent(llm, ToolRegistry(tools), RunLog([output]), max_tool_calls)
    return agent, transport, output


def tool_results(transport: ScriptedTransport) -> list:
    """Tool results the model saw in its last request."""
    return [json.loads(m["content"]) for m in transport.requests[-1]["messages"] if m["role"] == "tool"]


def test_runs_tool_then_returns_final_answer(task):
    agent, transport, output = make_agent([
        tool_call_reply(("calculator", {"expression": "2 + 2"})),
        final_reply("The answer is 4."),
    ])

    assert agent.run(task) == "The answer is 4."
    assert tool_results(transport) == ["4"]
    assert len(transport.requests) == 2
    assert output.getvalue().splitlines() == [
        "Calling LLM for next tool to invoke",
        "** Entering tool calculator **",
        "Parameter expression = 2 + 2",
        "** Exiting tool calculator **",
        "Calling LLM for next tool to invoke",
        "final response is = The answer is 4.",
    ]


def test_sends_tool_schemas_and_lists_tools_in_system_prompt(task):
    agent, transport, _ = make_agent([final_reply("done")])

    agent.run(task)

    request = transport.requests[0]
    assert request["tools"] == [CALCULATOR.schema]
    assert request["tool_choice"] == "auto"
    system_message = request["messages"][0]
    assert system_message["role"] == "system"
    assert "You have access to 1 tools: calculator." in system_message["content"]


def test_long_parameters_are_truncated_in_the_log(task):
    expression = "1 + " * 30 + "1"
    agent, _, output = make_agent([
        tool_call_reply(("calculator", {"expression": expression})),
        final_reply("31"),
    ])

    agent.run(task)

    assert f"Parameter expression = {expression[:50]}..." in output.getvalue().splitlines()


def test_tool_errors_are_returned_to_the_model(task):
    agent, transport, _ = make_agent([
        tool_call_reply(("calculator", {"expression": "1 / 0"})),
        final_reply("Cannot divide by zero."),
    ])

    assert agent.run(task) == "Cannot divide by zero."
    assert tool_results(transport) == [{"error": "ZeroDivisionError: division by zero"}]


def test_unknown_tools_and_bad_arguments_do_not_crash_the_run(task):
    agent, transport, output = make_agent([
        tool_call_reply(
            ("teleport", {}),
            ("calculator", "{not json"),
            ("calculator", "[1, 2]"),
            ("calculator", {"wrong_arg": "1"}),
        ),
        final_reply("Giving up."),
    ])

    assert agent.run(task) == "Giving up."
    unknown, bad_json, not_object, wrong_arg = tool_results(transport)
    assert "unknown tool 'teleport'" in unknown["error"]
    assert "not valid JSON" in bad_json["error"]
    assert "expected a JSON object" in not_object["error"]
    assert wrong_arg["error"].startswith("TypeError")
    # Only the call that actually reached a tool is logged as entering it.
    assert output.getvalue().count("** Entering tool") == 1


def test_stops_when_tool_call_cap_is_reached(task):
    agent, _, output = make_agent(
        [tool_call_reply(*[("calculator", {"expression": "1"})] * 3)],
        max_tool_calls=2,
    )

    assert agent.run(task) is None
    log = output.getvalue()
    assert log.count("** Entering tool calculator **") == 2
    assert log.splitlines()[-1] == "** TERMINATED: tool call cap reached **"


def test_stops_when_llm_call_cap_is_reached(task):
    agent, transport, output = make_agent(
        [tool_call_reply(("calculator", {"expression": "1"}))] * 5,
        max_llm_calls=2,
    )

    assert agent.run(task) is None
    assert len(transport.requests) == 2
    assert output.getvalue().splitlines()[-1] == "** TERMINATED: LLM call cap reached **"


def test_llm_calls_made_inside_tools_count_toward_the_cap(task):
    # No flag or registration needed: any tool that uses the LLM is capped automatically.
    transport = ScriptedTransport(tool_call_reply(("ask", {"question": "hi"})), final_reply("unused"))
    llm = LLM(transport, max_calls=1)
    ask_tool = Tool(
        "ask",
        "Asks the model a question.",
        {"type": "object", "properties": {"question": {"type": "string"}}},
        handler=lambda question: llm.complete([{"role": "user", "content": question}]).content,
    )
    output = io.StringIO()
    agent = Agent(llm, ToolRegistry([ask_tool]), RunLog([output]), max_tool_calls=20)

    assert agent.run(task) is None
    assert len(transport.requests) == 1
    assert output.getvalue().splitlines()[-1] == "** TERMINATED: LLM call cap reached **"


def test_rejected_request_is_retried_without_the_latest_tool_results(task):
    agent, transport, output = make_agent([
        tool_call_reply(("calculator", {"expression": "2 + 2"})),
        LLMRejected("content_filter", "The response was filtered."),
        final_reply("Recovered."),
    ])

    assert agent.run(task) == "Recovered."
    retried = transport.requests[-1]["messages"]
    assert retried[-2]["role"] == "tool"
    assert retried[-2]["content"].startswith("[removed:")
    assert retried[-1]["role"] == "user"
    assert "rejected the previous request (content_filter)" in retried[-1]["content"]
    assert "LLM request rejected (content_filter) = The response was filtered." in output.getvalue()


def test_repeated_rejection_stops_the_run_instead_of_using_up_the_cap(task):
    agent, transport, output = make_agent([
        tool_call_reply(("calculator", {"expression": "2 + 2"})),
        LLMRejected("context_length_exceeded", "Too long."),
        LLMRejected("context_length_exceeded", "Still too long."),
        final_reply("never sent"),
    ])

    assert agent.run(task) is None
    assert len(transport.requests) == 3
    assert output.getvalue().splitlines()[-1] == "** TERMINATED: LLM request rejected again after recovery **"


def test_oversized_tool_results_are_truncated(task):
    dump = Tool("dump", "Returns a lot.", {"type": "object", "properties": {}}, handler=lambda: "x" * 50_000)
    agent, transport, _ = make_agent([tool_call_reply(("dump", {})), final_reply("ok")], tools=(dump,))

    agent.run(task)

    [content] = [m["content"] for m in transport.requests[-1]["messages"] if m["role"] == "tool"]
    assert len(content) < 20_100
    assert content.endswith("characters]")
