import gc
import os
from pathlib import Path

import torch
import unsloth
from datasets import load_dataset
from PIL import Image
from trl import SFTConfig, SFTTrainer
from unsloth import FastVisionModel  # FastLanguageModel for LLMs
from unsloth import get_chat_template
from unsloth.trainer import UnslothVisionDataCollator

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def create_conversation(path_in):
    user_message = {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": Image.open(path_in),
            },
            {
                "type": "text",
                "text": "Recognize all the Japanese text in this image. For any kanji that has furigana above it, please format it in Markdown as `[漢字]{かんじ}`. Present the entire recognized text in this Markdown format. Return only the recognized text.\n",
            },
        ],
    }
    assistant_message = {
        "role": "assistant",
        "content": [{"type": "text", "text": "cat"}],
    }
    conversation = {"messages": [user_message, assistant_message]}
    return conversation


def main():
    torch.cuda.empty_cache()
    gc.collect()
    # instruction = "Write the LaTeX representation for this image."
    # dataset = load_dataset("unsloth/LaTeX_OCR", split="train")

    # def convert_to_conversation(sample):
    #     conversation = [
    #         {
    #             "role": "user",
    #             "content": [
    #                 {"type": "text", "text": instruction},
    #                 {"type": "image", "image": sample["image"]},
    #             ],
    #         },
    #         {
    #             "role": "assistant",
    #             "content": [{"type": "text", "text": sample["text"]}],
    #         },
    #     ]
    #     return {"messages": conversation}

    # For axolotl, keep image paths
    # for unsloth, load PIL image
    # cnt_val_samples
    converted_dataset = []
    for path_in in list(sorted(Path("./cropped").iterdir())):  # [:cnt_val_samples]
        converted_dataset.append(create_conversation(path_in))
        # = [ for i in range(100)]
    # converted_dataset = [convert_to_conversation(sample) for sample in dataset]

    print(converted_dataset[0])

    # return
    # model_name = "unsloth/Meta-Llama-3.1-8B-bnb-4bit"
    model_name = "unsloth/gemma-3-4b-it"
    # model_name = "unsloth/Llama-3.2-1B-Instruct"
    model, processor = FastVisionModel.from_pretrained(
        model_name=model_name,
        dtype=torch.float16,  # Use float16 for memory efficiency
        load_in_4bit=False,  # Use 4bit to reduce memory use. False for 16bit LoRA.
        load_in_8bit=True,
        # full_finetuning=True,
        use_gradient_checkpointing="unsloth",
        # use_gradient_checkpointing="unsloth",  # True or "unsloth" for long context
        # r=16,  # Add LoRA rank
        # lora_alpha=32,  # Add LoRA alpha
    )
    print("model loaded!")
    processor = get_chat_template(processor, "gemma-3")

    # model = FastVisionModel.for_training(model)
    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=True,  # False if not finetuning vision layers
        finetune_language_layers=True,  # False if not finetuning language layers
        finetune_attention_modules=True,  # False if not finetuning attention layers
        finetune_mlp_modules=True,  # False if not finetuning MLP layers
        r=16,  # The larger, the higher the accuracy, but might overfit
        lora_alpha=16,  # Recommended alpha == r at least
        lora_dropout=0,
        bias="none",
        random_state=3407,
        use_rslora=False,  # We support rank stabilized LoRA
        loftq_config=None,  # And LoftQ
        target_modules="all-linear",  # Optional now! Can specify a list if needed
        modules_to_save=[
            "lm_head",
            "embed_tokens",
        ],
    )
    print("model set for training!")

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
            max_steps=1000,
            fp16=True,  # Use mixed precision
            # num_train_epochs = 2,          # Set this instead of max_steps for full training runs
            learning_rate=2e-6,
            logging_steps=2,
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
            max_seq_length=4096,
        ),
    )
    # print("just in case check trainable params again:")
    # print(trainer.model.print_trainable_parameters())
    trainer_stats = trainer.train()
    print(trainer_stats)
    model.save_pretrained_merged(
        "merged_model",
        processor.tokenizer,
        save_method="merged_16bit",  # or "merged_4bit" for smaller size
    )


if __name__ == "__main__":
    main()
