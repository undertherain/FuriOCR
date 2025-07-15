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
prompt = "Recognize all the Japanese text in this image. For any kanji that has furigana above it, please format it in Markdown as `[漢字]{かんじ}`. Present the entire recognized text in this Markdown format. Return only the recognized text."


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

output_folder = Path("recognized_sloth")
output_folder.mkdir(exist_ok=True)
cnt_val_samples=10
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
    text_streamer = TextStreamer(tokenizer, skip_prompt=True, )

    # Generate the response from the model.
    outputs = model.generate(
        **inputs,
        # streamer=text_streamer,
        max_new_tokens=8192,
        use_cache=True,
        temperature=1,
    )
    decoded_output = tokenizer.batch_decode(outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]

    # Define the output file path.
    output_filename = f"{image_path.stem}.md"
    output_file_path = output_folder / output_filename

    # Save the decoded text to the file.
    output_file_path.write_text(decoded_output.strip())


print("\nInference complete.")
