"""Vision tool: reads a receipt/invoice image and returns its fields as JSON."""

import base64
import json
from pathlib import Path

from ..llm import LLM
from ..task import Workspace
from .base import Tool, strip_code_fences

PROMPT = (
    "Return ONLY a JSON object with keys: merchant (string), date (YYYY-MM-DD string), "
    "total (number), items (list of {name, price}). No markdown, no explanation."
)

IMAGE_MIME_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
}


def extract_from_image(image_path: Path, llm: LLM) -> dict:
    mime_type = IMAGE_MIME_TYPES.get(image_path.suffix.lower())
    if mime_type is None:
        raise ValueError(f"unsupported image type: {image_path.suffix}")

    encoded_image = base64.b64encode(image_path.read_bytes()).decode("ascii")
    reply = llm.complete([
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mime_type};base64,{encoded_image}",
                        "detail": "high",
                    },
                },
                {"type": "text", "text": PROMPT},
            ],
        },
    ])
    return json.loads(strip_code_fences(reply.content))


def extract_from_image_tool(llm: LLM, workspace: Workspace) -> Tool:
    return Tool(
        name="extract_from_image",
        description=(
            "Extract structured data from a receipt or invoice PNG/JPG image. "
            "Returns a dict with keys: merchant, date, total, and items. "
            "Use this for image resources that contain receipt or invoice details."
        ),
        parameters={
            "type": "object",
            "properties": {
                "image_path": {
                    "type": "string",
                    "description": "Filesystem path to the PNG or JPG image file.",
                },
            },
            "required": ["image_path"],
        },
        handler=lambda image_path: extract_from_image(workspace.resource_path(image_path), llm),
    )
