# -*- coding: utf-8 -*-
"""
A script to upload a locally saved, fully merged Hugging Face model and its
processor to the Hugging Face Hub.

UPDATED VERSION: Uses `upload_folder` and `ignore_patterns` to upload ONLY the
final model, skipping any intermediate checkpoint folders.
"""
from pathlib import Path
from huggingface_hub import HfApi, HfFolder, create_repo, upload_folder

# --- 1. CONFIGURE YOUR UPLOAD ---
# IMPORTANT: Make sure you have run `huggingface-cli login` in your terminal first.

# The local directory where your fine-tuned model is saved.
LOCAL_MODEL_PATH = "./lfm2-vl-furigana-ocr"

# The desired repository ID on the Hugging Face Hub.
HUB_REPO_ID = "blackbird/lfm2-vl-furigana-ocr" # <- CHANGE THIS

# --- 2. VERIFY LOCAL MODEL EXISTS ---
local_path = Path(LOCAL_MODEL_PATH)
if not local_path.exists() or not (local_path / "config.json").exists():
    print(f"❌ Error: Model directory not found or is invalid at '{LOCAL_MODEL_PATH}'")
    exit()
print(f"✅ Found local model at: {local_path.resolve()}")

# --- 3. CREATE REPO AND UPLOAD ---
try:
    print(f"\n🚀 Creating repository '{HUB_REPO_ID}' on the Hub...")
    create_repo(repo_id=HUB_REPO_ID, exist_ok=True, token=HfFolder.get_token())

    print(f"🌍 Uploading final model from '{LOCAL_MODEL_PATH}' (ignoring checkpoints)...")
    # Upload the entire folder, but ignore any checkpoint subdirectories.
    upload_folder(
        folder_path=LOCAL_MODEL_PATH,
        repo_id=HUB_REPO_ID,
        repo_type="model",
        commit_message="Upload final model",
        # This glob pattern tells the uploader to ignore any folder starting
        # with "checkpoint-" and all files/folders inside them.
        ignore_patterns=["checkpoint-*/**"], # <-- THIS IS THE KEY CHANGE
    )

    print("\n🎉 Success! Your final model has been uploaded.")
    print(f"   You can view it here: https://huggingface.co/{HUB_REPO_ID}")

except Exception as e:
    print(f"\n❌ An error occurred during upload: {e}")