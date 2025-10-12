import pathlib

from PIL import Image

# --- Configuration ---
# Set the paths to your folders here.
# The script assumes it is running in the parent directory of these folders.
FURIGANIZED_DIR = pathlib.Path("furiganized")
CROPPED_DIR = pathlib.Path("cropped")


def delete_empty_markdown_files():
    """
    Iterates through the furiganized directory and deletes any .md file
    that has a size of 0 bytes.
    """
    print(f"--- Checking for empty markdown files in '{FURIGANIZED_DIR}' ---")
    if not FURIGANIZED_DIR.is_dir():
        print(f"Error: Directory not found: '{FURIGANIZED_DIR}'")
        return

    # Use a list to avoid issues with iterating while deleting
    md_files = list(FURIGANIZED_DIR.glob("*.md"))
    deleted_count = 0

    for md_file in md_files:
        # Check if the file is empty
        if md_file.stat().st_size == 0:
            try:
                md_file.unlink()
                print(f"Deleted empty file: {md_file}")
                deleted_count += 1
            except OSError as e:
                print(f"Error deleting file {md_file}: {e}")

    print(f"Found and deleted {deleted_count} empty markdown file(s).\n")


def sync_and_clean_files():
    """
    Performs a two-way sync and cleanup:
    1. Deletes .png images in 'cropped' that don't have a corresponding .md file.
    2. Deletes both .png and .md files if the image is zero-size or corrupt.
    """
    print(
        f"--- Syncing '{CROPPED_DIR}' with '{FURIGANIZED_DIR}' and checking for corrupt images ---"
    )
    if not FURIGANIZED_DIR.is_dir():
        print(f"Error: Source directory not found: '{FURIGANIZED_DIR}'")
        return
    if not CROPPED_DIR.is_dir():
        print(f"Error: Image directory not found: '{CROPPED_DIR}'")
        return

    # Get a set of all markdown file stems (filename without extension) for fast lookups.
    md_stems = {p.stem for p in FURIGANIZED_DIR.glob("*.md")}

    # Use a list to avoid issues with iterating while deleting
    png_files = list(CROPPED_DIR.glob("*.png"))

    orphaned_deleted_count = 0
    corrupt_deleted_count = 0

    for png_file in png_files:
        # Check for orphaned images
        if png_file.stem not in md_stems:
            try:
                png_file.unlink()
                print(f"Deleted orphaned image: {png_file}")
                orphaned_deleted_count += 1
            except OSError as e:
                print(f"Error deleting orphaned image {png_file}: {e}")
            continue  # Move to the next file

        # Check for zero-size or corrupt images
        is_corrupt = False
        md_to_delete = FURIGANIZED_DIR / f"{png_file.stem}.md"

        if png_file.stat().st_size == 0:
            print(f"Found zero-size image: {png_file}")
            is_corrupt = True
        else:
            try:
                with Image.open(png_file) as img:
                    img.verify()  # Checks for file corruption without loading the full image
            except (IOError, SyntaxError) as e:
                print(f"Found corrupt image: {png_file} (Error: {e})")
                is_corrupt = True

        if is_corrupt:
            try:
                # Delete the corrupt PNG file
                png_file.unlink()
                print(f"  -> Deleted corrupt image: {png_file}")

                # Delete the corresponding MD file
                if md_to_delete.exists():
                    md_to_delete.unlink()
                    print(f"  -> Deleted corresponding markdown: {md_to_delete}")

                corrupt_deleted_count += 1
            except OSError as e:
                print(f"Error during cleanup for {png_file.stem}: {e}")

    print(f"Deleted {orphaned_deleted_count} orphaned image(s).")
    print(
        f"Found and cleaned {corrupt_deleted_count} corrupt image(s) and their markdown files.\n"
    )


def main():
    """
    Main function to run the cleanup process.
    """
    print("Starting file cleanup process...")

    # Step 1: Delete empty markdown files first.
    delete_empty_markdown_files()

    # Step 2: Sync folders and clean corrupt files.
    sync_and_clean_files()

    print("Cleanup process finished.")


if __name__ == "__main__":
    main()
