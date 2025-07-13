from pathlib import Path

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments


def create_simplified_finetuning_demo():
    """
    Debugs fine-tuning by using a very small model (smollm) and simplified
    text formatting to avoid potential chat template issues.
    """
    # 1. Define Model (Switched to a much smaller model for debugging)
    model_id = "google/gemma-3-4b-it"  # Commented out for now
    # model_id = "HuggingFaceTB/SmolLM2-135M-Instruct"

    # The input format is simplified to a plain string
    conversations = [{"user": "What is the Capital of France?", "assistant": "Paris"}]

    # 2. Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    # 3. Manually prepare data with simplified string formatting
    processed_data = []
    for i in range(128):
        # for chat in conversations:
        # We now create a simple string instead of using a complex chat template.
        # This is a more robust way to format input for debugging.
        chat = conversations[0]
        text = f"User: {chat['user']}\nAssistant: {chat['assistant']}"

        tokenized_output = tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=128,  # Reduced max_length for the smaller model
        )
        processed_data.append(
            {
                "input_ids": tokenized_output["input_ids"],
                "attention_mask": tokenized_output["attention_mask"],
                "labels": tokenized_output["input_ids"].copy(),
            }
        )

    # 4. Convert the list of dictionaries into a Hugging Face Dataset
    final_dataset = Dataset.from_list(processed_data)

    # 5. Load Model
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)
    model.resize_token_embeddings(len(tokenizer))

    # 6. Configure Training (Keeping stability settings)
    output_dir = Path("./smollm-finetuned-simplified")
    training_arguments = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=2,
        gradient_accumulation_steps=16,
        num_train_epochs=1,
        learning_rate=5e-7,
        max_grad_norm=0.1,
        logging_dir=str(output_dir / "logs"),
        logging_steps=1,
        use_cpu=True,
        report_to="none",
    )

    # 7. Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_arguments,
        train_dataset=final_dataset,
    )

    # 8. Start Fine-tuning
    print(f"Starting fine-tuning for small model '{model_id}' to debug...")
    trainer.train()
    print("Fine-tuning complete.")

    # 9. Save the final model
    final_model_path = output_dir / "final_model"
    trainer.save_model(str(final_model_path))
    print(f"Model saved to {final_model_path}")


if __name__ == "__main__":
    create_simplified_finetuning_demo()
