"""Text-to-SQL tool: asks the model to write a query for a described SQLite schema."""

from ..llm import LLM
from .base import Tool, strip_code_fences

PROMPT = (
    "Given this schema: {schema_description}\n"
    "Translate this request into SQL: {natural_language}\n"
    "Build SQL for one SQLite database at a time. Use only tables and columns "
    "that are available in the schema for the target database. If another "
    "database is needed, return a query for the current database only so the "
    "agent can use the result in a later tool call.\n"
    "Return ONLY the SQL query. No markdown fences, no explanation, no commentary."
)


def build_sql_query(natural_language: str, schema_description: str, llm: LLM) -> str:
    prompt = PROMPT.format(
        schema_description=schema_description,
        natural_language=natural_language,
    )
    reply = llm.complete([{"role": "user", "content": prompt}])
    return strip_code_fences(reply.content)


def build_sql_query_tool(llm: LLM) -> Tool:
    return Tool(
        name="build_sql_query",
        description=(
            "Translate a natural-language data request into a SQL query string. "
            "Provide the request and the relevant SQLite schema description, including "
            "table names and columns. Returns SQL only and does not execute it. "
            "Always pair this with execute_sql_query."
        ),
        parameters={
            "type": "object",
            "properties": {
                "natural_language": {
                    "type": "string",
                    "description": "What you want to compute or retrieve, written in English.",
                },
                "schema_description": {
                    "type": "string",
                    "description": (
                        "Description of the relevant SQLite table(s) and columns, "
                        "copied from the resource description in input.json."
                    ),
                },
            },
            "required": ["natural_language", "schema_description"],
        },
        handler=lambda natural_language, schema_description: build_sql_query(
            natural_language, schema_description, llm
        ),
        uses_llm=True,
    )
