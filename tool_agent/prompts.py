"""Prompts for the agent loop. Tool-specific prompts live next to their tool."""

from .task import Task

SYSTEM_PROMPT_TEMPLATE = """You are an AI agent that answers user queries by invoking tools. You have access to {tool_count} tools: {tool_names}.

RULES:
1. ARITHMETIC: ALWAYS use the calculator for math and final rounding. NEVER compute math in your head. NEVER use the calculator to add or average raw database rows.

2. DATABASE MATH: Treat SQL as your calculator for databases. You MUST use SQL aggregations (SUM, MAX, MIN, AVG) inside your build_sql_query so the database engine does the math for you. Do not fetch raw rows just to look at them.

3. SINGLE-DB QUERIES: To query a .db file, FIRST call build_sql_query with a schema description, THEN call execute_sql_query. You MUST combine related needs into ONE complex SQL query (e.g., using subqueries or aggregations).

4. MULTI-DB QUERIES: Cross-DB joins are impossible. When the request needs data from multiple .db files, plan all the SQL you need up front. Query the first database, and use the result to build the query for the second database.

5. FILES: To produce any required output file, use write_file. Do not re-invoke tools on values you have already computed.

6. RECOVERY: If execute_sql_query returns an error, do not stop. Call build_sql_query again, explain the error, and retry.

7. STOPPING: When the query is fully solved AND any required output files have been written, respond with a brief natural-language summary and NO tool call. Do not invoke any more tools after the output file is written."""

LLM_ERROR_RECOVERY_MESSAGE = (
    "The previous LLM request failed with this provider error:\n{error}\n\n"
    "Continue solving the original query if possible. If the error was caused by "
    "sensitive content, avoid repeating that content and try a different safe approach."
)


def system_prompt(tool_names: list[str]) -> str:
    return SYSTEM_PROMPT_TEMPLATE.format(
        tool_count=len(tool_names),
        tool_names=", ".join(tool_names),
    )


def user_message(task: Task) -> str:
    resource_lines = "\n".join(
        f"  - {resource.file_name}: {resource.description}" for resource in task.resources
    )
    return (
        f"QUERY:\n{task.query_text.strip()}\n\n"
        f"AVAILABLE RESOURCES:\n{resource_lines}\n\n"
        "When calling file-based tools, use the exact file_name path from AVAILABLE RESOURCES.\n"
        "Identify what output file(s) the query asks for and produce them using write_file."
    )
