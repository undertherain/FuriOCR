from pathlib import Path

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments


def main():
    """
    Creates and runs a simplified demo of fine-tuning a Gemma model.
    Data is prepared manually in a for loop without using map or a data collator.
    """
    # 1. Define Model and your conversations (Model updated)
    model_id = "google/gemma-3-4b-it"

    # Note: Fine-tuning a 4B parameter model on a CPU will be extremely slow
    # and require a significant amount of RAM (likely 16GB+).

    conversations = [
        [
            {"role": "user", "content": "What is the Capital of France?"},
            {"role": "model", "content": "Paris"},
        ]
    ]

    # 2. Load Tokenizer and set the chat template
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    chat_template = """{% for message in messages %}{% if message['role'] == 'user' %}{{ '<start_of_turn>user\n' + message['content'] + '<end_of_turn>\n' }}{% elif message['role'] == 'model' %}{{ '<start_of_turn>model\n' + message['content'] + '<end_of_turn>\n' }}{% endif %}{% endfor %}"""
    tokenizer.chat_template = chat_template
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    # 3. Manually prepare the data in a for loop
    processed_data = []
    for chat in conversations:
        formatted_chat = tokenizer.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=False
        )
        tokenized_output = tokenizer(
            formatted_chat,
            truncation=True,
            padding="max_length",
            max_length=512,
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

    # 6. Configure Training (Arguments updated)
    output_dir = Path("./gemma-finetuned-simplified")
    training_arguments = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=1,
        num_train_epochs=1,
        learning_rate=2e-5,  # Lower learning rate added
        logging_dir=str(output_dir / "logs"),
        logging_steps=1,
        use_cpu=True,  # Replaced 'no_cuda'
        report_to="none",
    )

    # 7. Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_arguments,
        train_dataset=final_dataset,
    )

    # 8. Start Fine-tuning
    print(f"Starting fine-tuning for model '{model_id}' on the CPU...")
    trainer.train()
    print("Fine-tuning complete.")

    # 9. Save the final model
    final_model_path = output_dir / "final_model"
    trainer.save_model(str(final_model_path))
    print(f"Model saved to {final_model_path}")


if __name__ == "__main__":
    main()
