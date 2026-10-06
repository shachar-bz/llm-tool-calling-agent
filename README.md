# Assignment #3 — Sample Files

This gist contains the sample input, query, resource files, and validation function
for Assignment #3 (Tool Calling / Function Calling).

## Files in this gist

| File                              | Purpose                                                  |
|-----------------------------------|----------------------------------------------------------|
| `input.json`                      | Describes the query name and the resource files          |
| `receipt_analysis.txt`            | The sample query your agent must answer                  |
| `receipt_analysis_validate.py`    | Validation function the grader uses to check correctness |
| `receipt.png`                     | Sample receipt image (Blue Moon Cafe, total $87.45)      |
| `orders.db`                       | SQLite database with 40 historical orders                |
| `customers.db`                    | SQLite database with 15 customer records                 |
| `prepare_dataset.py`              | Run once to verify all files are present                 |
| `README.md`                       | This file                                                |

## Setup

1. Download all 8 files from this gist into a single directory.
2. From that directory, run:
   ```
   python prepare_dataset.py
   ```
   You should see: `All 6 required files found. You are ready to start working on hw3.py.`

3. Place your `hw3.py` (your agent) in the same directory.
4. Run your agent with:
   ```
   python hw3.py
   ```

## What your `hw3.py` must do

Your agent must:

1. Read `input.json` to discover the query name (`receipt_analysis.txt`) and the resource files.
2. Read `receipt_analysis.txt` to get the actual query.
3. Initialize an OpenAI client using the Azure credentials in the assignment text.
4. Run a raw OpenAI tool-calling loop with 6 tools: `calculator`, `extract_from_image`,
   `build_sql_query`, `execute_sql_query`, `web_search`, `write_file`.
5. Let the LLM dynamically pick and chain the right tools to solve the query.
6. Write the result to `receipt_analysis_result.json` (the file the query asks for).
7. Log every step to `receipt_analysis.log` AND stdout in the required format.

You may NOT use LangGraph, LangChain agents, `create_react_agent`, or any other
prebuilt agent framework. See the assignment text for the full rubric.

## Expected correct answer

For the sample `receipt_analysis` query, the correct JSON answer is:

```json
{
  "merchant": "Blue Moon Cafe",
  "receipt_total": 87.45,
  "historical_total": 1240.30,
  "percentage_of_historical": 7.05,
  "top_customer_city": "Haifa"
}
```

The validation function checks each field with appropriate tolerances for floats
and is case-insensitive on strings.

## Important notes

- The grader will test your code with **different queries**, **different resource files**,
  and **different validation functions** that follow the same format. Do NOT hardcode
  anything specific to `receipt_analysis`, `Blue Moon Cafe`, the `orders` schema, or
  any specific column name.
- The query name's stem (without `.txt`) drives:
  - the log file name (`receipt_analysis.log`)
  - the validation file name (`receipt_analysis_validate.py`, with hyphens → underscores)
  - the validation function name (`receipt_analysis_answer`)
- Do NOT include the files from this gist with your submission. The grader will
  provide their own sample files when grading.
