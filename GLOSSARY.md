# LLM Tool-Calling Agent

An agent that answers a natural-language query by letting a language model chain tools over a set of files, within hard caps on how much work one run may do.

## Tasks

**Task**:
A query plus the resource files it may use. A task is the unit the agent is asked to solve.
_Avoid_: job, problem, assignment

**Manifest**:
The `input.json` file that describes a task: which file holds the query and which resources are available, each with a description.
_Avoid_: config, input file

**Query**:
The natural-language request a task asks the agent to fulfil, read from the file the manifest names.
_Avoid_: prompt, question

**Resource**:
A file declared in the manifest that tools may read, such as a receipt image or a SQLite database.
_Avoid_: asset, attachment, input

**Workspace**:
The folder holding a task's manifest, as tools see it: every path the model chooses must resolve inside it. The task's own files are read-only within it.
_Avoid_: root, sandbox, working directory

**Output file**:
A file the query asks the agent to produce, written into the workspace.
_Avoid_: result file, answer file

**Validator**:
A function that ships with a task and decides whether an output file is correct.
_Avoid_: grader, checker

## Runs

**Run**:
One attempt to solve a task. A run ends with a final answer, or is stopped by a call cap or a repeated rejection.
_Avoid_: session, execution

**Final answer**:
The model's reply that calls no tool, which ends a run successfully.
_Avoid_: final response, result

**Run log**:
The step-by-step record of a run, in a fixed line format, written to `<query>.log` and echoed to stdout.
_Avoid_: transcript, trace

**LLM-call cap**:
The most model requests one run may make, counting requests made inside tools as well as the agent's own.
_Avoid_: budget, quota

**Tool-call cap**:
The most tool calls one run may make.
_Avoid_: budget, quota

## Tools and the model

**Tool**:
A named capability the model may call, described to it by a schema: calculator, image extraction, SQL building, SQL execution, web search, file writing.
_Avoid_: function, action, plugin

**Tool call**:
One request by the model to run a tool with specific arguments.
_Avoid_: invocation, function call

**LLM-backed tool**:
A tool that itself asks the model for help, such as image extraction or SQL building. Its requests count toward the LLM-call cap.
_Avoid_: AI tool, smart tool

**Tool error**:
A failed tool call reported back to the model as an error, so it can correct itself and continue the run.
_Avoid_: exception, crash

**Transport**:
The channel that carries one request to the model provider and returns either its reply or a rejection.
_Avoid_: client, backend, connection

**Rejection**:
The provider refusing a request outright, for example because a content filter fired or the conversation is too long. Distinct from a tool error.
_Avoid_: bad request, API error
