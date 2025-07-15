from datasets import load_dataset
from PIL import Image
from transformers import AutoProcessor, AutoTokenizer
from unsloth import FastProcessor, get_chat_template

# Convert your dataset to the "role"/"content" format if necessary
from unsloth.chat_templates import standardize_sharegpt

# dataset = load_dataset("mlabonne/FineTome-100k", split="train")


# dataset = standardize_sharegpt(dataset)

# print(dataset[0])
# from unsloth.chat_templates import CHAT_TEMPLATES
# print(list(CHAT_TEMPLATES.keys()))


# tokenizer = AutoTokenizer.from_pretrained("unsloth/gemma-3-4b-it")
# tokenizer = get_chat_template(
#     tokenizer,
#     chat_template="gemma3",  # change this to the right chat_template name
# )
processor = AutoProcessor.from_pretrained("unsloth/gemma-3-4b-it")

print(type(tokenizer))
convo = {
    "role": "user",
    "content": [
        {
            "type": "text",
            "text": "hi",
        },
        {
            "type": "image",
            "image": Image.open("./cat.jpg"),
        },
    ],
}
conversation = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello, who are you?"},
    {"role": "assistant", "content": "I am a large language model created by Google."},
]

# tokenized_chat = tokenizer.apply_chat_template(
# conversation, tokenize=False, add_generation_prompt=False
# )

# processor.
# print(tokenized_chat)
# from transformers import AutoTokenizer

# Replace "google/gemma-2-9b-it" with your desired model
# tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-9b-it")
#
# res = tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False)
# print(res)
# processor = get_chat_template(processor, "gemma-3")
# print(processor)
