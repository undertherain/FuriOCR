from email import message
from pathlib import Path

import torch
from datasets import load_dataset
from PIL import Image
from trl import SFTConfig, SFTTrainer
from unsloth import FastVisionModel  # FastLanguageModel for LLMs
from unsloth import get_chat_template
from unsloth.trainer import UnslothVisionDataCollator

# dataset = load_dataset("unsloth/LaTeX_OCR", split="train")

instruction = "Write the LaTeX representation for this image."


# def convert_to_conversation(sample):
#     conversation = [
#         {
#             "role": "user",
#             "content": [
#                 {"type": "text", "text": instruction},
#                 {"type": "image", "image": sample["image"]},
#             ],
#         },
#         {"role": "assistant", "content": [{"type": "text", "text": sample["text"]}]},
#     ]
#     return {"messages": conversation}


# For axolotl, keep image paths
# for unsloth, load PIL image
def create_conversation():
    user_message = {
        "role": "user",
        "content": [
            {"type": "image", "path": Image.open("./cat.jpg")},
            {
                "type": "text",
                "text": "Recognize the main object in this image. Respond with one word.",
            },
        ],
    }
    assistant_message = {
        "role": "assistant",
        "content": [{"type": "text", "text": "cat"}],
    }
    conversation = {"messages": [user_message, assistant_message]}
    return conversation


converted_dataset = [create_conversation() for i in range(100)]
# converted_dataset = [convert_to_conversation(sample) for sample in dataset]


print(converted_dataset[0])

processor = get_chat_template(processor, "gemma-3")

model, processor = FastVisionModel.from_pretrained(
    "unsloth/gemma-3-4b-pt",
    load_in_4bit=False,  # Use 4bit to reduce memory use. False for 16bit LoRA.
    use_gradient_checkpointing="unsloth",  # True or "unsloth" for long context
)


FastVisionModel.for_training(model)  # Enable for training!

trainer = SFTTrainer(
    model=model,
    train_dataset=converted_dataset,
    processing_class=processor.tokenizer,
    data_collator=UnslothVisionDataCollator(model, processor),
    args=SFTConfig(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        gradient_checkpointing=True,
        # use reentrant checkpointing
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_grad_norm=0.3,  # max gradient norm based on QLoRA paper
        warmup_ratio=0.03,
        max_steps=30,
        # num_train_epochs = 2,          # Set this instead of max_steps for full training runs
        learning_rate=2e-4,
        logging_steps=1,
        save_strategy="steps",
        save_safetensors=False,
        optim="adamw_torch_fused",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        seed=3407,
        output_dir="outputs",
        report_to="none",  # For Weights and Biases
        # You MUST put the below items for vision finetuning:
        remove_unused_columns=False,
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},
        max_seq_length=2048,
    ),
)
trainer_stats = trainer.train()
