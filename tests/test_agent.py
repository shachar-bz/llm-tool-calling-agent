import io
import json

from fakes import ScriptedLLM, final_reply, tool_call_reply

from tool_agent.agent import Agent
from tool_agent.budget import Budget
from tool_agent.run_log import RunLog
from tool_agent.tools import Tool, ToolRegistry
from tool_agent.tools.calculator import CALCULATOR


def make_agent(replies, tools=(CALCULATOR,), max_llm_calls=20, max_tool_calls=20):
    budget = Budget(max_llm_calls, max_tool_calls)
    llm = ScriptedLLM(budget, replies)
    output = io.StringIO()
    agent = Agent(llm, ToolRegistry(tools), budget, RunLog([output]))
    return agent, llm, budget, output


def tool_results(llm: ScriptedLLM) -> list:
    """Tool results the model saw in its last request."""
    return [json.loads(m["content"]) for m in llm.requests[-1] if m["role"] == "tool"]


def test_runs_tool_then_returns_final_answer(task):
    agent, llm, budget, output = make_agent([
        tool_call_reply(("calculator", {"expression": "2 + 2"})),
        final_reply("The answer is 4."),
    ])

    assert agent.run(task) == "The answer is 4."
    assert tool_results(llm) == ["4"]
    assert (budget.llm_calls, budget.tool_calls) == (2, 1)
    assert output.getvalue().splitlines() == [
        "Calling LLM for next tool to invoke",
        "** Entering tool calculator **",
        "Parameter expression = 2 + 2",
        "** Exiting tool calculator **",
        "Calling LLM for next tool to invoke",
        "final response is = The answer is 4.",
    ]


def test_system_prompt_lists_registered_tools(task):
    agent, llm, _, _ = make_agent([final_reply("done")])

    agent.run(task)

    system_message = llm.requests[0][0]
    assert system_message["role"] == "system"
    assert "You have access to 1 tools: calculator." in system_message["content"]


def test_long_parameters_are_truncated_in_the_log(task):
    expression = "1 + " * 30 + "1"
    agent, _, _, output = make_agent([
        tool_call_reply(("calculator", {"expression": expression})),
        final_reply("31"),
    ])

    agent.run(task)

    assert f"Parameter expression = {expression[:50]}..." in output.getvalue().splitlines()


def test_tool_errors_are_returned_to_the_model(task):
    agent, llm, _, _ = make_agent([
        tool_call_reply(("calculator", {"expression": "1 / 0"})),
        final_reply("Cannot divide by zero."),
    ])

    assert agent.run(task) == "Cannot divide by zero."
    assert tool_results(llm) == [{"error": "ZeroDivisionError: division by zero"}]


def test_unknown_tools_and_bad_arguments_do_not_crash_the_run(task):
    agent, llm, budget, _ = make_agent([
        tool_call_reply(
            ("teleport", {}),
            ("calculator", "{not json"),
            ("calculator", "[1, 2]"),
            ("calculator", {"wrong_arg": "1"}),
        ),
        final_reply("Giving up."),
    ])

    assert agent.run(task) == "Giving up."
    unknown, bad_json, not_object, wrong_arg = tool_results(llm)
    assert "unknown tool 'teleport'" in unknown["error"]
    assert "not valid JSON" in bad_json["error"]
    assert "expected a JSON object" in not_object["error"]
    assert wrong_arg["error"].startswith("TypeError")
    assert budget.tool_calls == 1  # only the call that actually reached a tool


def test_stops_when_tool_call_cap_is_reached(task):
    agent, _, budget, output = make_agent(
        [tool_call_reply(*[("calculator", {"expression": "1"})] * 3)],
        max_tool_calls=2,
    )

    assert agent.run(task) is None
    assert budget.tool_calls == 2
    assert output.getvalue().splitlines()[-1] == "** TERMINATED: tool call cap reached **"


def test_stops_when_llm_call_cap_is_reached(task):
    agent, _, budget, output = make_agent(
        [tool_call_reply(("calculator", {"expression": "1"}))] * 5,
        max_llm_calls=2,
    )

    assert agent.run(task) is None
    assert budget.llm_calls == 2
    assert output.getvalue().splitlines()[-1] == "** TERMINATED: LLM call cap reached **"


def test_llm_backed_tool_is_not_run_without_llm_budget(task):
    ran = []
    llm_tool = Tool("vision", "Looks at images.", {"type": "object", "properties": {}},
                    handler=lambda: ran.append(True), uses_llm=True)
    agent, _, _, output = make_agent(
        [tool_call_reply(("vision", {}))],
        tools=(llm_tool,),
        max_llm_calls=1,
    )

    assert agent.run(task) is None
    assert ran == []
    assert output.getvalue().splitlines()[-1] == "** TERMINATED: LLM call cap reached **"
