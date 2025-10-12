from os import path
from pathlib import Path

from html2image import Html2Image

# from selenium import webdriver

# browser = webdriver.Firefox()
# html_file_path = "file:///home/blackbird/Projects_heavy/AI/FuriOCR/htmls/0490b1e3-408a-4e7c-9f45-4065819de537.html"
# html_file_path = Path("./htmls/0490b1e3-408a-4e7c-9f45-4065819de537.html")
# html_file_path = "/home/blackbird/Projects_heavy/AI/FuriOCR/htmls/0490b1e3-408a-4e7c-9f45-4065819de537.html"
# browser.get(html_file_path)
# browser.save_screenshot("screenie.png")

hti = Html2Image(
    custom_flags=["--default-background-color=FFFFFF"], output_path="./pngs"
)
# html_file_path = "input.html"
for html_file_path in Path("./htmls").iterdir():
    # output_png_path = Path("./pngs") / html_file_path.name
    hti.screenshot(
        url=html_file_path.as_posix(),
        save_as=html_file_path.with_suffix(".png").name,
        # output_path="pngs",
    )
# hti.screenshot(url="https://www.python.org", save_as="python_org.png")
# print(f"Screenshot saved successfully to '{output_png_path}'")
