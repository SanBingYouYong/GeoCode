from PIL import Image


def resize_and_convert(img_path: str) -> None:
    # Open an image
    image = Image.open(img_path)
    # Resize the image (e.g., to 300x300 pixels)
    resized_image = image.resize((224, 224))
    # Convert the image to grayscale
    grayscale_image = resized_image.convert('L')
    # Save the grayscale image
    grayscale_image.save(img_path)
    print(f"PIL saved img to {img_path}")