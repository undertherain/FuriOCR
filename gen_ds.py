import json

user_message = {
    "role": "user",
    "content": [
        {"type": "image", "path": "./cat.jpg"},
        {
            "type": "text",
            "text": "Recognize the main object in this image. Respond with one word.",
        },
    ],
}
assistant_message = {"role": "assistant", "content": [{"type": "text", "text": "cat"}]}
messages = {"messages": [user_message, assistant_message]}
with open("dummy.jsonl", "w") as f:
    for i in range(100):
        f.write(json.dumps(messages))
        f.write("\n")
