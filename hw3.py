# Authors: Shachar Ben Zur, Ron Isakov

from openai import BadRequestError, OpenAI
from Tools.build_sql_query import BUILD_SQL_QUERY_SCHEMA, build_sql_query_with_client
from Tools.calculator import CALCULATOR_SCHEMA, calculator
from Tools.execute_sql_query import EXECUTE_SQL_QUERY_SCHEMA, execute_sql_query
from Tools.extract_from_image import EXTRACT_FROM_IMAGE_SCHEMA, extract_from_image_with_client
from Tools.web_search import WEB_SEARCH_SCHEMA, web_search
from Tools.write_file import WRITE_FILE_SCHEMA, write_file
import json, os, sys

DEPLOYMENT_NAME = "gpt-4.1-mini"
ENDPOINT = "https://softwareengineeringusingai.openai.azure.com/openai/v1/"
API_KEY = os.environ["AZURE_OPENAI_API_KEY"]

client = OpenAI(
api_key=API_KEY,
base_url=ENDPOINT,
)

INPUT_JSON_PATH = "input.json"
LOG_VALUE_PREVIEW_LENGTH = 50
LOG_VALUE_SUFFIX = "..."

SYSTEM_PROMPT = """You are an AI agent that answers user queries by invoking tools. You have access to 6 tools: calculator, extract_from_image, build_sql_query, execute_sql_query, web_search, write_file.

RULES:
1. ARITHMETIC: ALWAYS use the calculator for math and final rounding. NEVER compute math in your head. NEVER use the calculator to add or average raw database rows.

2. DATABASE MATH: Treat SQL as your calculator for databases. You MUST use SQL aggregations (SUM, MAX, MIN, AVG) inside your build_sql_query so the database engine does the math for you. Do not fetch raw rows just to look at them.

3. SINGLE-DB QUERIES: To query a .db file, FIRST call build_sql_query with a schema description, THEN call execute_sql_query. You MUST combine related needs into ONE complex SQL query (e.g., using subqueries or aggregations).

4. MULTI-DB QUERIES: Cross-DB joins are impossible. When the request needs data from multiple .db files, plan all the SQL you need up front. Query the first database, and use the result to build the query for the second database.

5. FILES: To produce any required output file, use write_file. Do not re-invoke tools on values you have already computed.

6. RECOVERY: If execute_sql_query returns an error, do not stop. Call build_sql_query again, explain the error, and retry.

7. STOPPING: When the query is fully solved AND any required output files have been written, respond with a brief natural-language summary and NO tool call. Do not invoke any more tools after the output file is written."""

LLM_CAP = 20
TOOL_CAP = 20
LLM_BACKED_TOOLS = {"extract_from_image", "build_sql_query"}
TOOL_FILE_ARGUMENT_KEYS = {
    "extract_from_image": ("image_path",),
    "execute_sql_query": ("db_path",),
}

CONTENT_FILTER_RECOVERY_MESSAGE = (
    "The previous LLM request failed with this provider error:\n{error}\n\n"
    "Continue solving the original query if possible. If the error was caused by "
    "sensitive content, avoid repeating that content and try a different safe approach."
)

llm_calls = 0
tool_calls = 0
LOG_FILE = None
RESOURCE_FILE_PATHS = {}


def setup_logging(query_name):
    global LOG_FILE
    stem = os.path.splitext(os.path.basename(query_name))[0]
    subdir = os.path.dirname(query_name)
    log_path = os.path.join(subdir, stem + ".log") if subdir else stem + ".log"

    if subdir:
        os.makedirs(subdir, exist_ok=True)

    if LOG_FILE is not None:
        LOG_FILE.close()

    LOG_FILE = open(log_path, "w", encoding="utf-8")


def log(msg):
    msg = str(msg)
    print(msg)
    sys.stdout.flush()
    if LOG_FILE is not None:
        LOG_FILE.write(msg + "\n")
        LOG_FILE.flush()


def log_tool_entry(tool_name, tool_args):
    log(f"** Entering tool {tool_name} **")
    for key, value in tool_args.items():
        value_preview = str(value)
        if len(value_preview) > LOG_VALUE_PREVIEW_LENGTH:
            value_preview = value_preview[:LOG_VALUE_PREVIEW_LENGTH] + LOG_VALUE_SUFFIX
        log(f"Parameter {key} = {value_preview}")


def log_tool_exit(tool_name):
    log(f"** Exiting tool {tool_name} **")


def build_user_message(query_text: str, resources: list) -> str:
    resource_lines = "\n".join(
        f"  - {r['file_name']}: {r['description']}" for r in resources
    )
    return (
        f"QUERY:\n{query_text.strip()}\n\n"
        f"AVAILABLE RESOURCES:\n{resource_lines}\n\n"
        "When calling file-based tools, use the exact file_name path from AVAILABLE RESOURCES.\n"
        "Identify what output file(s) the query asks for and produce them using write_file."
    )


def build_resource_file_paths(resources: list) -> dict:
    exact_paths = {}
    basename_counts = {}

    for resource in resources:
        file_path = resource.get("file_name")
        if not file_path:
            continue
        basename = os.path.basename(file_path)
        exact_paths[file_path] = file_path
        basename_counts[basename] = basename_counts.get(basename, 0) + 1

    for resource in resources:
        file_path = resource.get("file_name")
        if not file_path:
            continue
        basename = os.path.basename(file_path)
        if basename_counts[basename] == 1:
            exact_paths[basename] = file_path

    return exact_paths


def resolve_tool_args(tool_name: str, tool_args: dict) -> dict:
    resolved_args = dict(tool_args)

    for key in TOOL_FILE_ARGUMENT_KEYS.get(tool_name, ()):
        value = resolved_args.get(key)
        if not isinstance(value, str):
            continue

        if value in RESOURCE_FILE_PATHS:
            resolved_args[key] = RESOURCE_FILE_PATHS[value]
            continue

        basename = os.path.basename(value)
        if basename in RESOURCE_FILE_PATHS:
            resolved_args[key] = RESOURCE_FILE_PATHS[basename]

    return resolved_args


def extract_from_image(image_path: str) -> dict:
    global llm_calls
    llm_calls += 1
    return extract_from_image_with_client(image_path, client, DEPLOYMENT_NAME)


def build_sql_query(natural_language: str, schema_description: str) -> str:
    global llm_calls
    llm_calls += 1
    return build_sql_query_with_client(
        natural_language,
        schema_description,
        client,
        DEPLOYMENT_NAME,
    )


TOOL_SCHEMAS = [
    CALCULATOR_SCHEMA,
    EXTRACT_FROM_IMAGE_SCHEMA,
    BUILD_SQL_QUERY_SCHEMA,
    EXECUTE_SQL_QUERY_SCHEMA,
    WEB_SEARCH_SCHEMA,
    WRITE_FILE_SCHEMA,
]

TOOL_DISPATCH = {
    "calculator": calculator,
    "extract_from_image": extract_from_image,
    "build_sql_query": build_sql_query,
    "execute_sql_query": execute_sql_query,
    "web_search": web_search,
    "write_file": write_file,
}


def main():
    global llm_calls, tool_calls, LOG_FILE, RESOURCE_FILE_PATHS

    try:
        with open(INPUT_JSON_PATH) as f:
            config = json.load(f)
        query_name = config["query_name"]
        resources = config["resources"]
        RESOURCE_FILE_PATHS = build_resource_file_paths(resources)

        setup_logging(query_name)

        with open(query_name) as f:
            query_text = f.read()

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_message(query_text, resources)},
        ]

        while True:
            if llm_calls >= LLM_CAP:
                log("** TERMINATED: LLM call cap reached **")
                break

            log("Calling LLM for next tool to invoke")
            llm_calls += 1
            try:
                response = client.chat.completions.create(
                    model=DEPLOYMENT_NAME,
                    messages=messages,
                    tools=TOOL_SCHEMAS,
                    tool_choice="auto",
                )
            except BadRequestError as exception:
                error_message = f"{type(exception).__name__}: {exception}"
                log(f"LLM BadRequestError = {error_message}")
                messages.append({
                    "role": "user",
                    "content": CONTENT_FILTER_RECOVERY_MESSAGE.format(
                        error=error_message,
                    ),
                })
                continue

            msg = response.choices[0].message
            messages.append(msg.model_dump(exclude_none=True))

            if not msg.tool_calls:
                log(f"final response is = {msg.content}")
                break

            cap_hit = False
            for tc in msg.tool_calls:
                if tool_calls >= TOOL_CAP:
                    log("** TERMINATED: tool call cap reached **")
                    cap_hit = True
                    break

                tool_name = tc.function.name
                tool_args = resolve_tool_args(
                    tool_name,
                    json.loads(tc.function.arguments),
                )

                if tool_name in LLM_BACKED_TOOLS and llm_calls >= LLM_CAP:
                    log("** TERMINATED: LLM call cap reached **")
                    cap_hit = True
                    break

                log_tool_entry(tool_name, tool_args)
                result = TOOL_DISPATCH[tool_name](**tool_args)
                log_tool_exit(tool_name)
                tool_calls += 1

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(result, default=str),
                })

            if cap_hit:
                break
    finally:
        if LOG_FILE is not None:
            LOG_FILE.close()
            LOG_FILE = None


if __name__ == "__main__":
    main()
