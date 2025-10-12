# -*- coding: utf-8 -*-
"""
Fine-tunes LFM2-VL on a custom Japanese OCR dataset with furigana.
Data is expected in two parallel folders:
- ./cropped: Contains PNG images.
- ./furiganized: Contains corresponding .md files with transcriptions.
"""
import torch
import transformers
import trl
from pathlib import Path
from PIL import Image
from datasets import Dataset, DatasetDict
from transformers import AutoModelForImageTextToText, AutoProcessor, TrainingArguments
import jiwer
# Verify packages are installed correctly
print(f"📦 PyTorch version: {torch.__version__}")
print(f"🤗 Transformers version: {transformers.__version__}")
print(f"📊 TRL version: {trl.__version__}")

# --- 1. Model Loading (Unchanged from original script) ---
# This section loads the pre-trained model and processor.
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

#   model_id = "LiquidAI/LFM2-VL-450M"
model_id = "LiquidAI/LFM2-VL-1.6B"

print("📚 Loading processor...")
processor = AutoProcessor.from_pretrained(
    model_id,
    trust_remote_code=True,
    max_image_tokens=256,
)

print("🧠 Loading model...")
model = AutoModelForImageTextToText.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto",
)

print("\n✅ Local model loaded successfully!")
print(f"📖 Vocab size: {len(processor.tokenizer)}")
print(f"🖼️ Image processed in up to {processor.max_tiles} patches of size {processor.tile_size}")
print(f"🔢 Parameters: {model.num_parameters():,}")
print(f"💾 Model size: ~{model.num_parameters() * 2 / 1e9:.1f} GB (bfloat16)")


# --- 2. Custom Dataset Loading and Preprocessing (MODIFIED SECTION) ---
# This section is adapted to load your local image and text files.

def load_custom_dataset(image_dir: Path, text_dir: Path):
    """Loads image paths and their corresponding text from local directories."""
    image_files = list(image_dir.glob("*.png"))
    data = []
    print(f"Found {len(image_files)} images in {image_dir}.")
    
    for img_path in image_files:
        text_path = text_dir / img_path.with_suffix(".md").name
        if text_path.exists():
            # We read the text content now. The image will be loaded on-the-fly.
            ground_truth_text = text_path.read_text(encoding="utf-8").strip()
            data.append({"image_path": str(img_path), "gt_answer": ground_truth_text})
        else:
            print(f"Warning: Missing annotation for {img_path.name}, skipping.")
            
    return Dataset.from_list(data)

# Define paths and load the data
image_dir = Path("./cropped")
text_dir = Path("./furiganized")
full_dataset = load_custom_dataset(image_dir, text_dir)

# Split the dataset into training and evaluation sets
split_dataset = full_dataset.train_test_split(test_size=0.1, seed=42) # Using a 90/10 split
train_dataset = split_dataset["train"]
eval_dataset = split_dataset["test"]

print("\n✅ Custom SFT Dataset loaded:")
print(f"   📚 Train samples: {len(train_dataset)}")
print(f"   🧪 Eval samples: {len(eval_dataset)}")

# Define the conversation structure for the model
system_message = (
    "You are an expert OCR model specializing in recognizing Japanese text, "
    "including handling furigana annotations correctly."
)
user_prompt = (
    "Recognize all the Japanese text in this image. For any kanji that has furigana "
    "above it, please format it in Markdown as `[漢字]{かんじ}`. Present the entire "
    "recognized text in this Markdown format. Return only the recognized text."
)

def format_ocr_sample(sample):
    """Formats a single data sample into the required chat template structure."""
    # Load the image on-the-fly
    image = Image.open(sample["image_path"]).convert("RGB")
    
    return [
        {"role": "system", "content": [{"type": "text", "text": system_message}]},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": user_prompt},
            ],
        },
        {"role": "assistant", "content": [{"type": "text", "text": sample["gt_answer"]}]},
    ]

# Apply the formatting to the datasets
train_dataset = [format_ocr_sample(s) for s in train_dataset]
eval_dataset = [format_ocr_sample(s) for s in eval_dataset]

print("\n✅ SFT Dataset formatted:")
print(f"   📚 Train samples: {len(train_dataset)}")
print(f"   🧪 Eval samples: {len(eval_dataset)}")
print(f"\n📝 Example formatted sample prompt: {train_dataset[0][1]['content'][1]['text']}")


# --- 3. Collate Function (Unchanged from original script) ---
# This function batches the data correctly for the model.
def create_collate_fn(processor):
    """Create a collate function that prepares batch inputs for the processor."""
    def collate_fn(sample):
        batch = processor.apply_chat_template(sample, tokenize=True, return_dict=True, return_tensors="pt")
        labels = batch["input_ids"].clone()
        if processor.tokenizer.pad_token_id is not None:
            labels[labels == processor.tokenizer.pad_token_id] = -100
        batch["labels"] = labels
        return batch
    return collate_fn

collate_fn = create_collate_fn(processor)


# --- 4. PEFT/LoRA Configuration (Unchanged from original script) ---
# Using LoRA for efficient fine-tuning.
from peft import LoraConfig, get_peft_model

target_modules = [
    "q_proj", "v_proj", "fc1", "fc2", "linear",
    "gate_proj", "up_proj", "down_proj",
]

peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.05,
    r=8,
    bias="none",
    target_modules=target_modules,
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()


# --- 5. Training Configuration and Execution (Minor changes) ---
# SFTConfig is slightly adapted for the new dataset.
from trl import SFTConfig, SFTTrainer

# Define a new output directory for your fine-tuned model
output_dir = "lfm2-vl-furigana-ocr"

sft_config = SFTConfig(
    output_dir=output_dir,
    num_train_epochs=2,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    learning_rate=5e-5, # Slightly lower learning rate can be more stable
    warmup_ratio=0.1,
    weight_decay=0.01,
    logging_steps=10,
    optim="adamw_torch_8bit",
    gradient_checkpointing=True,
    max_length=1024, # Max sequence length
    dataset_kwargs={"skip_prepare_dataset": True},
    report_to=None, # Set to "wandb" or "tensorboard" if you want to log
    #evaluation_strategy="steps", # Uncomment to evaluate during training
    #eval_steps=50,               # Uncomment to set evaluation frequency
)


# training_args = TrainingArguments(
#     output_dir=output_dir,
#     # Training parameters
#     num_train_epochs=3,
#     per_device_train_batch_size=1,
#     gradient_accumulation_steps=16,
#     learning_rate=5e-5,
#     warmup_ratio=0.1,
#     weight_decay=0.01,
#     optim="adamw_torch_8bit",
#     gradient_checkpointing=True,
#     # Evaluation and logging
#     evaluation_strategy="steps",
#     eval_steps=50,
#     logging_strategy="steps",
#     logging_steps=10,
#     # Saving
#     save_strategy="steps",
#     save_steps=100,
#     save_total_limit=2,
#     load_best_model_at_end=True,
#     metric_for_best_model="eval_loss",
#     greater_is_better=False,
#     # Reporting
#     report_to="none",
# )

def compute_metrics(eval_preds):
    """Computes Character Error Rate (CER) from model predictions."""
    logits, labels = eval_preds
    # The predictions are the logits, so we need to take the argmax to get the token IDs
    predictions = np.argmax(logits[0], axis=-1)

    # Decode predicted tokens and labels
    decoded_preds = processor.batch_decode(predictions, skip_special_tokens=True)
    
    # Replace -100 in labels as we can't decode them
    labels[labels == -100] = processor.tokenizer.pad_token_id
    decoded_labels = processor.batch_decode(labels, skip_special_tokens=True)

    # Calculate CER
    cer = jiwer.cer(decoded_labels, decoded_preds)
    
    return {"cer": cer}

print("🏗️  Creating SFT trainer...")
sft_trainer = SFTTrainer(
    model=model,
    args=sft_config,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=collate_fn,
    compute_metrics=compute_metrics,
    # The tokenizer from the processor is used for chat templating
    # peft_config=peft_config, # Pass peft_config if not already applied
)

print("\n🚀 Starting SFT training...")
sft_trainer.train()

print("🎉 SFT training completed!")

# Save the final LoRA adapter
sft_trainer.save_model()
print(f"💾 LoRA adapter saved to: {sft_config.output_dir}")


# --- 6. (Optional) Merge and Save Full Model (Unchanged from original script) ---
# This combines the LoRA weights with the base model for easier deployment.
if hasattr(model, 'peft_config'):
    print("\n🔄 Merging LoRA weights...")
    # merge adapter layers with base model and save
    model = model.merge_and_unload()

# Save the fully merged model and the processor
model.save_pretrained(output_dir)
processor.save_pretrained(output_dir)
print(f"💾 Full model and processor saved to: {output_dir}")