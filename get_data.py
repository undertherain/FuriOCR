# get_data.py
# Description: This script downloads entries from the Japanese subset of the
# HuggingFaceFW/fineweb-edu dataset, segments them into smaller chunks
# by sentence, truncates them, and saves them as individual files.

from pathlib import Path

from datasets import load_dataset


def download_and_segment_fineweb(num_entries=100):
    """
    Downloads, segments, and saves a subset of the fineweb-edu-japanese dataset.

    Each entry is split into sentences using "。" as a delimiter. Each sentence
    is truncated to 100 characters and saved to a unique file.

    Args:
        num_entries (int): The number of entries to download from the dataset.
    """
    try:
        print("Attempting to load the dataset via streaming...")
        dataset = load_dataset(
            "hotchpotch/fineweb-2-edu-japanese",
            name="sample_10BT",
            streaming=True,
            split="train",
        )

        print(
            f"Successfully loaded. Fetching and processing the first {num_entries} entries..."
        )
        subset = dataset.take(num_entries)
        output_dir = Path("inputs")

        # Process each entry from the dataset stream.
        entry_count = 0
        for entry in subset:
            entry_count += 1
            entry_id = entry.get("id", f"unknown_id_{entry_count}")
            text_content = entry.get("text", "")

            print(f"\n--- Processing Entry {entry_count}: {entry_id} ---")

            # Segment the text by sentences and filter out empty ones.
            sentences = [
                s.strip() for s in text_content.split("entry_id。") if s.strip()
            ]

            if not sentences:
                print("    No content to save.")
                continue

            # Save each segmented and truncated sentence to a new file.
            for i, sentence in enumerate(sentences):
                # Re-add the period and truncate to the specified length.
                processed_sentence = (sentence + "。")[:100]

                output_filename = f"{entry_id[10:-1]}_{i}"
                output_path = output_dir / output_filename

                try:
                    # Ensure files are written with UTF-8 encoding for Japanese text.
                    with open(output_path, "w", encoding="utf-8") as f:
                        f.write(processed_sentence)
                except OSError as e:
                    print(f"    Error writing file {output_filename}: {e}")

            print(f"    Saved {len(sentences)} segments.")

        print(f"\nSuccessfully processed {entry_count} entries.")

    except Exception as e:
        print(f"An error occurred: {e}")
        print(
            "Please ensure you have an active internet connection and the 'datasets' library is installed."
        )


if __name__ == "__main__":
    # Create the output directory if it doesn't exist.
    Path("inputs").mkdir(exist_ok=True)
    download_and_segment_fineweb(1010)
