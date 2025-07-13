from pathlib import Path

import torch
from datasets import Dataset
from PIL import Image
from transformers import (
    AutoProcessor,
    Gemma3ForConditionalGeneration,
    Trainer,
    TrainingArguments,
)

# def create_dummy_cat_image():
#     """Creates a simple black image with a white square and saves it as cat.jpg."""
#     if not Path("cat.jpg").exists():
#         img = Image.new("RGB", (224, 224), color="black")
#         pixels = img.load()
#         for i in range(50, 150):
#             for j in range(50, 150):
#                 pixels[i, j] = (255, 255, 255)
#         img.save("cat.jpg")


def main():
    # 1. Create a dummy image
    # create_dummy_cat_image()
    image_path = Path("cat.jpg")
    image = Image.open(image_path)

    # 2. Define the dummy dataset
    prompt = "what is the main item in the image"
    assistant_reply = "cat"

    # Replicate the sample 100 times
    dataset_data = [
        {"prompt": prompt, "image": image.copy(), "answer": assistant_reply}
        for _ in range(100)
    ]

    # Create a Hugging Face Dataset
    dummy_dataset = Dataset.from_list(dataset_data)

    # 3. Load model and processor
    model_id = "google/gemma-3-4b-it"
    processor = AutoProcessor.from_pretrained(model_id)
    model = Gemma3ForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float32,  # Use float32 for CPU
    )
    print("Model loaded!")
    # 4. Preprocess the dataset
    def preprocess_function(examples):
        # Create the chat template
        chats = [
            [
                {"role": "user", "content": [f"<image>\n{p}"]},
                {"role": "assistant", "content": a},
            ]
            for p, a in zip(examples["prompt"], examples["answer"])
        ]

        # Apply the chat template and tokenize
        inputs = processor.tokenizer.apply_chat_template(
            chats,
            tokenize=True,
            add_generation_prompt=False,
            padding="max_length",
            max_length=2048,
            return_tensors="pt",
        )

        # Preprocess the images
        raw_images = [img.convert("RGB") for img in examples["image"]]
        image_inputs = processor(images=raw_images, return_tensors="pt")

        inputs["pixel_values"] = image_inputs.pixel_values

        return inputs

    processed_dataset = dummy_dataset.map(preprocess_function, batched=True)
    print("DataSet preprocessed")
    # 5. Configure Training Arguments for CPU
    training_args = TrainingArguments(
        output_dir="./gemma-4b-multimodal-finetuned",
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=1e-4,
        logging_dir="./logs",
        logging_steps=10,
        no_cuda=True,  # Explicitly disable CUDA
        use_cpu=True,  # Explicitly use CPU
    )

    # 6. Define the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=processed_dataset,
    )

    # 7. Start fine-tuning
    print("Starting fine-tuning on CPU...")
    trainer.train()
    print("Fine-tuning complete.")

    # 8. Save the fine-tuned model
    model.save_pretrained("./gemma-4b-multimodal-finetuned")
    processor.save_pretrained("./gemma-4b-multimodal-finetuned")
    print("Model and processor saved.")


if __name__ == "__main__":
    main()
