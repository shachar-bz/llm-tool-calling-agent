import os

WRITE_FILE_SCHEMA = {
    "type": "function",
    "function": {
        "name": "write_file",
        "description": (
            "Write a string to a file, creating parent directories if needed. "
            "Use this to produce the output file or files requested by the query."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "file_content": {
                    "type": "string",
                    "description": "The exact bytes/text to write.",
                },
                "file_name": {
                    "type": "string",
                    "description": "Filesystem path of the file to write.",
                },
            },
            "required": ["file_content", "file_name"],
        },
    },
}


def write_file(file_content: str, file_name: str) -> str:
    parent_directory = os.path.dirname(file_name)
    if parent_directory:
        os.makedirs(parent_directory, exist_ok=True)

    with open(file_name, "w", encoding="utf-8") as output_file:
        output_file.write(file_content)

    return f"Wrote {len(file_content.encode())} bytes to {file_name}"
