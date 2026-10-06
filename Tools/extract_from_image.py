import base64
import json
import os

EXTRACT_FROM_IMAGE_PROMPT = (
    "Return ONLY a JSON object with keys: merchant (string), date (YYYY-MM-DD string), "
    "total (number), items (list of {name, price}). No markdown, no explanation."
)

IMAGE_MIME_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
}

EXTRACT_FROM_IMAGE_SCHEMA = {
    "type": "function",
    "function": {
        "name": "extract_from_image",
        "description": (
            "Extract structured data from a receipt or invoice PNG/JPG image. "
            "Returns a dict with keys: merchant, date, total, and items. "
            "Use this for image resources that contain receipt or invoice details."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "image_path": {
                    "type": "string",
                    "description": "Filesystem path to the PNG or JPG image file.",
                },
            },
            "required": ["image_path"],
        },
    },
}


def strip_json_fences(response_text):
    response_text = response_text.strip()
    if response_text.startswith("```"):
        lines = response_text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        response_text = "\n".join(lines).strip()
    return response_text


def extract_from_image_with_client(image_path: str, client, deployment_name: str) -> dict:
    try:
        extension = os.path.splitext(image_path)[1].lower()
        mime_type = IMAGE_MIME_TYPES.get(extension)
        if mime_type is None:
            return {"error": f"Unsupported image type: {extension}"}

        with open(image_path, "rb") as image_file:
            encoded_bytes = base64.b64encode(image_file.read()).decode("utf-8")

        response = client.chat.completions.create(
            model=deployment_name,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime_type};base64,{encoded_bytes}",
                                "detail": "high",
                            },
                        },
                        {
                            "type": "text",
                            "text": EXTRACT_FROM_IMAGE_PROMPT,
                        },
                    ],
                },
            ],
        )
        response_text = response.choices[0].message.content
        return json.loads(strip_json_fences(response_text))
    except Exception as exception:
        return {"error": str(exception)}
