from pathlib import Path
from turtle import width

from PIL import Image, ImageChops


def crop_margins():
    """
    Scans a directory for PNG images, crops the white margins,
    and saves the processed files to a new directory.
    """
    # --- Configuration ---
    # The folder containing your original .png files.
    input_folder_name = "pngs"
    # The folder where cropped images will be saved.
    output_folder_name = "cropped"
    # The desired margin in pixels around the content.
    margin_pixels = 2

    # --- Script ---
    current_directory = Path.cwd()
    input_dir = current_directory / input_folder_name
    output_dir = current_directory / output_folder_name

    # Create the output directory if it doesn't already exist.
    output_dir.mkdir(exist_ok=True)

    print(f"Starting image processing...")
    print(f"Input folder: {input_dir}")
    print(f"Output folder: {output_dir}\n")

    # Find all .png files in the input directory.
    image_files = list(input_dir.glob("*.png"))

    if not image_files:
        print(f"Error: No .png files found in the '{input_folder_name}' directory.")
        print(
            "Please create an 'input' folder, place your PNGs inside, and run the script again."
        )
        return

    processed_count = 0
    for image_path in image_files:
        try:
            with Image.open(image_path) as img:
                # Convert the image to grayscale for accurate bounding box detection.
                # Then, invert the colors. The content becomes white, and the background black.
                img = img.crop((0, 0, img.width - 20, img.height))
                inverted_img = ImageChops.invert(img.convert("L"))

                # getbbox() finds the bounding box of all non-black pixels in the inverted image.
                # This corresponds to the content area of the original image.
                bbox = inverted_img.getbbox()

                if bbox:
                    # The bbox is a tuple: (left, upper, right, lower).
                    # We adjust it to add the desired margin.
                    # We also ensure the new box doesn't go outside the original image dimensions.
                    left = max(0, bbox[0] - margin_pixels)
                    upper = max(0, bbox[1] - margin_pixels)
                    right = min(img.width, bbox[2] + margin_pixels)
                    lower = min(img.height, bbox[3] + margin_pixels)

                    # Crop the original image using the new bounding box.
                    cropped_img = img.crop((left, upper, right, lower))

                    # Construct the full path for the output file.
                    output_path = output_dir / image_path.name
                    cropped_img.save(output_path)
                    print(
                        f"[SUCCESS] Cropped '{image_path.name}' and saved to '{output_path.relative_to(current_directory)}'"
                    )
                    processed_count += 1
                else:
                    # This case handles images that are entirely white or empty.
                    print(f"[SKIPPED] No content found in '{image_path.name}'.")

        except Exception as e:
            print(f"[ERROR] Could not process '{image_path.name}'. Reason: {e}")

    print(
        f"\nProcessing complete. Successfully processed {processed_count} of {len(image_files)} images."
    )


if __name__ == "__main__":
    crop_margins()
