import sys
from pathlib import Path

import torch
from PIL import Image
from transformers import TextStreamer
from unsloth import FastVisionModel

# 1. Configuration
# -----------------
# Set the path to your saved Unsloth vision model directory.
model_path = sys.argv[1]
output_folder = Path("recognized_sloth") / Path(model_path).stem
output_folder.mkdir(exist_ok=True)
# Set the path to the folder containing your images.
image_folder_path = "cropped"
# The prompt to be used for each image.
prompt = "Recognize all the Japanese text in this image. For any kanji that has furigana above it, please format it in Markdown as `[漢字]{かんじ}`. Present the entire recognized text in this Markdown format. Return only the recognized text."


# 2. Load Model and Tokenizer
# ---------------------------
# Load the fine-tuned model and tokenizer from the specified path.
# We're using load_in_4bit=True for memory efficiency.
try:
    model, processor = FastVisionModel.from_pretrained(
        model_name=model_path,
        dtype=None,
        load_in_4bit=True,
    )
except Exception as e:
    print(f"Error loading the model: {e}")
    print(
        "Please ensure the 'model_path' is correct and points to a valid Unsloth model directory."
    )
    exit()

# Enable fast inference mode.
print("processor:", type(processor))
FastVisionModel.for_inference(model)

cnt_val_samples = 10
for image_path in list(sorted(Path("./cropped").iterdir()))[:cnt_val_samples]:
    print("\n## processing file", image_path)
    # print(f"\nProcessing image: {image_path.name}")
    # print("-" * 30)

    image = Image.open(image_path)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    input_text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    # Tokenize the image and text prompt together.
    inputs = processor(
        images=[image],
        text=[input_text],
        return_tensors="pt",
    ).to("cuda")

    # Use a text streamer for cleaner output.
    text_streamer = TextStreamer(
        processor,
        skip_prompt=True,
    )

    # Generate the response from the model.
    outputs = model.generate(
        **inputs,
        # streamer=text_streamer,
        max_new_tokens=8192,
        use_cache=True,
        temperature=0.1,
    )
    decoded_output = processor.batch_decode(
        outputs[:, inputs.input_ids.shape[1] :], skip_special_tokens=True
    )[0]

    # Define the output file path.
    output_filename = f"{image_path.stem}.md"
    output_file_path = output_folder / output_filename

    # Save the decoded text to the file.
    print(decoded_output)
    output_file_path.write_text(decoded_output.strip())


print("\nInference complete.")
