import os
import subprocess

# import sys


def convert_md_to_pdf(directory):
    """
    Converts all Markdown files in a given directory to PDF format.

    Args:
        directory (str): The path to the directory containing the .md files.
    """
    if not os.path.isdir(directory):
        print(f"Error: Directory not found at '{directory}'")
        return

    for filename in os.listdir(directory):
        if filename.endswith(".md"):
            md_path = os.path.join(directory, filename)
            pdf_path = os.path.splitext(md_path)[0] + ".pdf"

            try:
                print(f"Converting {md_path} to {pdf_path}...")
                subprocess.run(
                    ["pandoc", md_path, "-o", pdf_path, "--defaults", "md_style.yaml"],
                    check=True,
                )
                print(f"Successfully converted {filename}")
            except subprocess.CalledProcessError as e:
                print(f"Error converting {filename}: {e}")
            except FileNotFoundError:
                print("Error: 'md2pdf' command not found.")
                print("Please ensure md2pdf is installed and in your system's PATH.")
                break


if __name__ == "__main__":
    # if len(sys.argv) != 2:
    #     print("Usage: python make_pdfs.py <directory>")
    # else:
    #     convert_md_to_pdf(sys.argv[1])
    convert_md_to_pdf("furiganized")
