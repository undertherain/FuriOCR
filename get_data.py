# main.py
# Description: This script downloads the first 100 entries from the Japanese subset
# of the HuggingFaceFW/fineweb-edu dataset using the datasets library.

# Ensure you have the 'datasets' library installed:
# pip install datasets

from pathlib import Path

from datasets import load_dataset


def download_fineweb_subset(num_entries=100):
    """
    Downloads and prints a subset of the HuggingFaceFW/fineweb-edu dataset.

    Args:
        num_entries (int): The number of entries to download.
    """
    try:
        print("Attempting to load the dataset...")
        # Load the dataset in streaming mode to avoid downloading the full dataset.
        # We specify the 'ja' (Japanese) configuration of the dataset.
        dataset = load_dataset(
            "hotchpotch/fineweb-2-edu-japanese",
            name="sample_10BT",
            streaming=True,
            split="train",
        )

        print(f"Successfully loaded. Fetching the first {num_entries} entries...")

        # Take the first 'num_entries' from the stream
        subset = dataset.take(num_entries)

        # Iterate over the subset and print each entry
        count = 0
        for entry in subset:
            print(f"\n--- Entry {count + 1} ---")
            entry_id = entry.get("id", "N/A")
            print(f"ID: {entry_id}")
            print(f"URL: {entry.get('url', 'N/A')}")
            print(f"Dump: {entry.get('dump', 'N/A')}")
            # The text content can be quite long, so we'll print a snippet.
            text_snippet = entry.get("text", "")[:500]
            print(f"Text (first 500 chars): {text_snippet}...")
            with open(Path("inputs") / entry_id[10:-1], "w") as f:
                f.write(entry.get("text", ""))
            count += 1

        print(f"\nSuccessfully downloaded and printed {count} entries.")

    except Exception as e:
        print(f"An error occurred: {e}")
        print(
            "Please ensure you have an active internet connection and the 'datasets' library is installed correctly."
        )
        print("You can install it with: pip install datasets")


if __name__ == "__main__":
    download_fineweb_subset(100)
