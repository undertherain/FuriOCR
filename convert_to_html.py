import re
from pathlib import Path


def convert_furigana_to_html(path_in, path_out, template_path):
    """
    Reads a Markdown file, converts furigana syntax to HTML <ruby> tags,
    and injects it into an HTML template.

    The script specifically looks for the pattern [Kanji]{kana} and converts it to:
    <ruby><rb>Kanji</rb><rt>kana</rt></ruby>

    Other Markdown elements like tables, titles, etc., are treated as plain text
    and are not converted.

    Args:
        md_file_path (str): The path to the input Markdown file.
        template_path (str): The path to the HTML template file.
    """
    try:
        # 1. Read the content of the input Markdown file
        with open(path_in, "r", encoding="utf-8") as f:
            markdown_text = f.read()

        # 2. Define the regex pattern to find furigana: [Kanji]{kana}
        #    - \[([^\]]+)\] captures the Kanji part (the base text).
        #    - \{([^\}]+)\} captures the kana part (the ruby text).
        furigana_pattern = r"\[([^\]]+)\]\{([^\}]+)\}"

        # 3. Define the replacement format using HTML <ruby> tags.
        #    - \1 refers to the first captured group (Kanji).
        #    - \2 refers to the second captured group (kana).
        replacement_html = r"<ruby><rb>\1</rb><rt>\2</rt></ruby>"

        # 4. Perform the substitution on the entire text
        converted_text = re.sub(furigana_pattern, replacement_html, markdown_text)

        # 5. Read the HTML template
        with open(template_path, "r", encoding="utf-8") as f:
            template_html = f.read()

        # 6. Inject the converted content into the template
        final_html = template_html.replace("{{content}}", converted_text)

        # 7. Determine the output file path
        #    - It will be the same as the input file, but with an .html extension.

        # 8. Write the final HTML to the output file
        with open(path_out, "w", encoding="utf-8") as f:
            f.write(final_html)

        # print(f"Successfully converted '{md_file_path}' to '{output_file_path}'")

    except FileNotFoundError as e:
        print(f"Error: {e}. Please check your file paths.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":

    # md_file = "input.md"
    Path("htmls").mkdir(exist_ok=True)
    template_file = "template.html"
    for path_in in Path("./furiganized").iterdir():
        print("## processing", path_in)
        path_out = Path("htmls") / path_in.with_suffix(".html").name
        convert_furigana_to_html(path_in, path_out, template_file)
