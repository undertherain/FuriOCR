import argparse
import pathlib

import textdistance


def calculate_cer(s1: str, s2: str) -> float:
    """
    Calculates the Character Error Rate (CER) between two strings.
    CER is defined as the Levenshtein distance divided by the length of the reference string.
    """
    # Normalize to handle different line endings and strip whitespace
    s1 = s1.strip().replace("\r\n", "\n")
    s2 = s2.strip().replace("\r\n", "\n")

    if not s1:
        return (
            1.0 if s2 else 0.0
        )  # If reference is empty, error is 100% if prediction is not

    distance = textdistance.levenshtein(s1, s2)
    return distance / len(s1)


def analyze_accuracy(original_dir: pathlib.Path, recognized_dir: pathlib.Path):
    """
    Compares markdown files in two directories and reports the average CER.

    Args:
        original_dir: Path to the directory with the ground truth markdown files.
        recognized_dir: Path to the directory with the OCR model's output files.
    """
    print(f"Comparing files in '{original_dir}' with '{recognized_dir}'\n")

    original_files = {p.name: p for p in original_dir.glob("*.md")}
    recognized_files = {p.name: p for p in recognized_dir.glob("*.md")}

    common_files = sorted(list(original_files.keys() & recognized_files.keys()))

    if not common_files:
        print("Error: No matching markdown files found in both directories.")
        print(f"Files in original: {list(original_files.keys())}")
        print(f"Files in recognized: {list(recognized_files.keys())}")
        return

    total_cer = 0
    results = []

    for filename in common_files:
        original_path = original_files[filename]
        recognized_path = recognized_files[filename]

        try:
            original_text = original_path.read_text(encoding="utf-8")
            recognized_text = recognized_path.read_text(encoding="utf-8")

            cer = calculate_cer(original_text, recognized_text)
            total_cer += cer
            results.append((filename, cer))

        except Exception as e:
            print(f"Could not process file {filename}: {e}")

    print("-" * 50)
    print(f"{'Filename':<30} | {'Character Error Rate (CER)':>20}")
    print("-" * 50)

    for filename, cer in results:
        print(f"{filename:<30} | {f'{cer:.2%}':>20}")

    print("-" * 50)

    if results:
        average_cer = total_cer / len(results)
        print(f"\n{'Average CER:':<30} | {f'{average_cer:.2%}':>20}")
        print(f"{'Overall Accuracy:':<30} | {f'{1 - average_cer:.2%}':>20}")
    else:
        print("No files were successfully processed.")


if __name__ == "__main__":
    # --- Setup for running the script ---
    # Create dummy directories and files for demonstration purposes.
    # In a real scenario, these directories would already exist.
    original_dir = pathlib.Path("./furiganized")
    recognized_dir = pathlib.Path("./recognized/gemma3_12b")

    # original_dir.mkdir(exist_ok=True)
    # recognized_dir.mkdir(exist_ok=True)

    # # Sample 1: Perfect match
    # (original_dir / "sample1.md").write_text("これは[日本語]{にほんご}のテキストです。", encoding='utf-8')
    # (recognized_dir / "sample1.md").write_text("これは[日本語]{にほんご}のテキストです。", encoding='utf-8')

    # # Sample 2: One character error in furigana
    # (original_dir / "sample2.md").write_text("彼女は[速]{はや}く[走]{はし}る。", encoding='utf-8')
    # (recognized_dir / "sample2.md").write_text("彼女は[速]{はか}く[走]{はし}る。", encoding='utf-8') # はや -> はか

    # # Sample 3: Missing furigana and a kanji error
    # (original_dir / "sample3.md").write_text("[今日]{きょう}は[天気]{てんき}がいいですね。", encoding='utf-8')
    # (recognized_dir / "sample3.md").write_text("[今日]は[元気]{てんき}がいいですね。", encoding='utf-8') # 天 -> 元 and missing furigana

    # # Sample 4: An extra file in one directory to show it's ignored
    # (original_dir / "extra.md").write_text("このファイルは無視されます。", encoding='utf-8')

    # # --- Running the analysis ---
    # You can run this script from your terminal.
    # Example: python your_script_name.py --original ./original --recognized ./recognized
    parser = argparse.ArgumentParser(
        description="Calculate Character Error Rate for OCR output."
    )
    parser.add_argument(
        "--original",
        type=pathlib.Path,
        default=original_dir,
        help="Directory with original markdown files.",
    )
    parser.add_argument(
        "--recognized",
        type=pathlib.Path,
        default=recognized_dir,
        help="Directory with recognized markdown files.",
    )

    args = parser.parse_args()

    analyze_accuracy(args.original, args.recognized)
