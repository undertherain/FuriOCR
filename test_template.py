from datasets import load_dataset
from PIL import Image
from transformers import AutoProcessor, AutoTokenizer

# Convert your dataset to the "role"/"content" format if necessary
# from unsloth.chat_templates import standardize_sharegpt

# from unsloth import get_chat_template


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

# print(type(tokenizer))
# image = Image.open("./cropped/0111663f-1f37-4762-9bd3-67e173cbbf0c.png")
image = Image.open("./cropped/seta.png")
convo = {
    "role": "user",
    "content": [
        {
            "type": "text",
            "text": "what is in this image?",
        },
        {
            "type": "image",
            "image": image,
        },
    ],
}
# conversation = [
#     {"role": "system", "content": "You are a helpful assistant."},
#     {"role": "user", "content": "Hello, who are you?"},
#     {"role": "assistant", "content": "I am a large language model created by Google."},
# ]

input_text = processor.tokenizer.apply_chat_template(
    [convo],
    tokenize=False,
    add_generation_prompt=True,
)
print("applied chat template as follows:")
print(input_text)
inputs = processor(
    images=[image],
    text=[input_text],
    return_tensors="pt",
)

print("tokenized_inputs:")
print(inputs.pixel_values.shape)

# Process the image and text together
# prompt = "<image>\nUSER: What's in this image?\nASSISTANT:"
# inputs = processor(
# text=prompt,
# images=convo["content"][1]["image"],
# return_tensors="pt",
# )

# The number of image "tokens" is the length of the second dimension of the pixel_values tensor.
# This represents the number of patches the image was divided into.
# num_image_tokens = inputs["pixel_values"].shape[1]

# print(f"The image was converted into {num_image_tokens} tokens.")

# tokenized_chat = tokenizer.apply_chat_template(
# conversation, tokenize=False, add_generation_prompt=False
# )

# processor.de
# print(tokenized_chat)
# from transformers import AutoTokenizer

# Replace "google/gemma-2-9b-it" with your desired model
# tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-9b-it")
#
# res = tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False)
# print(res)
# processor = get_chat_template(processor, "gemma-3")
# print(processor)
