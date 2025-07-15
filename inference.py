import torch
from unsloth import FastVisionModel
from PIL import Image
from pathlib import Path
from transformers import TextStreamer

# 1. Configuration
# -----------------
# Set the path to your saved Unsloth vision model directory.
model_path = "merged_model"
# Set the path to the folder containing your images.
image_folder_path = "cropped"
# The prompt to be used for each image.
prompt = "Recognize all the Japanese text in this image. For any kanji that has furigana above it, please format it in Markdown as `[漢字]{かんじ}`. Present the entire recognized text in this Markdown format. Return only the recognized text.\n"


# 2. Load Model and Tokenizer
# ---------------------------
# Load the fine-tuned model and tokenizer from the specified path.
# We're using load_in_4bit=True for memory efficiency.
try:
    model, tokenizer = FastVisionModel.from_pretrained(
        model_name=model_path,
        dtype=None,
        load_in_4bit=True,
    )
except Exception as e:
    print(f"Error loading the model: {e}")
    print("Please ensure the 'model_path' is correct and points to a valid Unsloth model directory.")
    exit()

# Enable fast inference mode.
FastVisionModel.for_inference(model)


# 3. Prepare Image and Prompt
# ---------------------------
# Get a list of all image files in the specified folder.
# This supports common image formats.
image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
try:
    image_paths = [p for p in Path(image_folder_path).glob('**/*') if p.suffix.lower() in image_extensions]
    if not image_paths:
        print(f"No images found in the directory: {image_folder_path}")
        exit()
except FileNotFoundError:
    print(f"The image directory was not found: {image_folder_path}")
    exit()


# 4. Perform Inference
# --------------------
for image_path in image_paths:
    try:
        print(f"\nProcessing image: {image_path.name}")
        print("-" * 30)

        # Open the image file.
        image = Image.open(image_path)

        # Format the prompt for the model.
        # The chat template expects a list of messages.
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        # Apply the chat template to format the input.
        input_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # Tokenize the image and text prompt together.
        inputs = tokenizer(
            [image],
            [input_text],
            return_tensors="pt",
        ).to("cuda")

        # Use a text streamer for cleaner output.
        text_streamer = TextStreamer(tokenizer, skip_prompt=True)

        # Generate the response from the model.
        _ = model.generate(
            **inputs,
            streamer=text_streamer,
            max_new_tokens=128,
            use_cache=True,
        )

    except Exception as e:
        print(f"Could not process image {image_path.name}. Error: {e}")

print("\nInference complete.")
