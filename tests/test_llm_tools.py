"""The LLM-backed tools, driven through the real LLM with a scripted transport."""

from pathlib import Path

import pytest
from support import ScriptedTransport, final_reply

from tool_agent.llm import LLM
from tool_agent.tools.build_sql_query import build_sql_query
from tool_agent.tools.extract_from_image import extract_from_image


def llm_replying(*texts: str | None) -> tuple[LLM, ScriptedTransport]:
    transport = ScriptedTransport(*(final_reply(text) for text in texts))
    return LLM(transport, max_calls=10), transport


def test_build_sql_query_sends_schema_and_request_and_strips_fences():
    llm, transport = llm_replying("```sql\nSELECT SUM(amount) FROM orders\n```")

    sql = build_sql_query("total amount", "orders(amount REAL)", llm)

    assert sql == "SELECT SUM(amount) FROM orders"
    prompt = transport.requests[0]["messages"][0]["content"]
    assert "orders(amount REAL)" in prompt
    assert "total amount" in prompt


@pytest.mark.parametrize("reply", [None, "", "```sql\n```"])
def test_build_sql_query_rejects_an_empty_reply(reply):
    llm, _ = llm_replying(reply)

    with pytest.raises(ValueError, match="no SQL"):
        build_sql_query("total", "orders(amount REAL)", llm)


@pytest.fixture
def receipt(tmp_path: Path) -> Path:
    path = tmp_path / "receipt.png"
    path.write_bytes(b"\x89PNG\r\n\x1a\n not really an image")
    return path


def test_extract_from_image_sends_the_image_and_parses_json(receipt: Path):
    llm, transport = llm_replying('```json\n{"merchant": "Cafe", "total": 12.5}\n```')

    assert extract_from_image(receipt, llm) == {"merchant": "Cafe", "total": 12.5}
    image_part = transport.requests[0]["messages"][0]["content"][0]
    assert image_part["image_url"]["url"].startswith("data:image/png;base64,")


@pytest.mark.parametrize(("reply", "error"), [(None, ValueError), ("[1, 2]", ValueError), ("not json", ValueError)])
def test_extract_from_image_rejects_replies_that_are_not_a_json_object(receipt: Path, reply, error):
    llm, _ = llm_replying(reply)

    with pytest.raises(error):
        extract_from_image(receipt, llm)


def test_extract_from_image_rejects_unsupported_file_types(tmp_path: Path):
    llm, transport = llm_replying()

    with pytest.raises(ValueError, match="unsupported image type"):
        extract_from_image(tmp_path / "receipt.gif", llm)
    assert transport.requests == []
