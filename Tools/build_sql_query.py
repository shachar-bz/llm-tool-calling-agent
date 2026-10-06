BUILD_SQL_QUERY_PROMPT = (
    "Given this schema: {schema_description}\n"
    "Translate this request into SQL: {natural_language}\n"
    "Build SQL for one SQLite database at a time. Use only tables and columns "
    "that are available in the schema for the target database. If another "
    "database is needed, return a query for the current database only so the "
    "agent can use the result in a later tool call.\n"
    "Return ONLY the SQL query. No markdown fences, no explanation, no commentary."
)

BUILD_SQL_QUERY_SCHEMA = {
    "type": "function",
    "function": {
        "name": "build_sql_query",
        "description": (
            "Translate a natural-language data request into a SQL query string. "
            "Provide the request and the relevant SQLite schema description, including "
            "table names and columns. Returns SQL only and does not execute it. "
            "Always pair this with execute_sql_query."
        ),
        "parameters": {
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
    },
}


def clean_sql_response(response_text):
    return (
        response_text.strip()
        .replace("```sql", "")
        .replace("```SQL", "")
        .replace("```", "")
        .strip()
    )


def build_sql_query_with_client(
    natural_language: str,
    schema_description: str,
    client,
    deployment_name: str,
):
    try:
        prompt = BUILD_SQL_QUERY_PROMPT.format(
            schema_description=schema_description,
            natural_language=natural_language,
        )
        response = client.chat.completions.create(
            model=deployment_name,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
        )
        response_text = response.choices[0].message.content
        return clean_sql_response(response_text)
    except Exception as exception:
        return {"error": str(exception)}
