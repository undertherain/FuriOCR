import base64
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

# --- Configuration ---
# It is recommended to set your OpenAI API key as an environment variable
# for security purposes.
# You can get your key from https://platform.openai.com/api-keys
load_dotenv()
client = OpenAI(
    # api_key=os.environ.get("OPENAI_API_KEY"),
    # base_url="http://127.0.0.1:2600/v1",
    base_url="http://ai-a100-01.r-ccs27.riken.jp:11434/v1",
)
# MODEL = "gpt-4.1"  # Or another suitable multimodal model
# MODEL = "gemma3:4b"
MODEL = "furi"


def encode_image(image_path):
    """Encodes the image at the given path to a base64 string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def recognize_japanese_text_with_furigana(image_path):
    """
    Sends an image to the OpenAI API and requests text recognition with furigana.

    Args:
      image_path: The path to the image file.

    Returns:
      The recognized text in Markdown format with furigana, or an error message.
    """
    if not os.path.exists(image_path):
        raise RuntimeError("Error: Image file not found at the specified path.")

    base64_image = encode_image(image_path)
    print("len image:", len(base64_image))

    # try:
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Recognize all the Japanese text in this image. For any kanji that has furigana above it, please format it in Markdown as `[漢字]{かんじ}`. Present the entire recognized text in this Markdown format. Return only the recognized text.\n",
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{base64_image}"},
                    },
                ],
            }
        ],
        max_tokens=4096,
    )
    return response.choices[0].message.content
    # except Exception as e:
    #   return f"An error occurred: {e}"


if __name__ == "__main__":
    # path_in = "./cropped/seta.png"  # Replace with the actual path to your image
    path_dst = Path("recognized") / MODEL.replace(":", "_")
    path_dst.mkdir(exist_ok=True, parents=True)
    cnt_val_samples = 10
    for path_in in list(sorted(Path("./cropped").iterdir()))[:cnt_val_samples]:
        print("\n## processing file", path_in)
        recognized_text = recognize_japanese_text_with_furigana(path_in)
        path_out = path_dst / path_in.with_suffix(".md").name
        with open(path_out, "w") as f:
            f.write(recognized_text)
