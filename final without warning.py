import io
import face_recognition
import numpy as np
from PIL import Image, ImageDraw
import warnings  # Import the warnings module

RED = (255, 0, 0)
GREEN = (0, 255, 0)
SIZE = 500
HALF_SIZE = 250

class EncodingError(Exception):
    pass

def get_concat_h_blank(im1, im2, color=(0, 0, 0)):
    dst = Image.new("RGB", (im1.width + im2.width, max(im1.height, im2.height)), color)
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst

def features(data):
    img = Image.open(io.BytesIO(data))
    img = img.convert("RGB")

    w, h = img.size
    # Use LANCZOS resampling instead of ANTIALIAS to avoid the DeprecationWarning
    img = img.resize((SIZE, int(SIZE * (h / w))), Image.LANCZOS)

    img_np = np.array(img)
    face_encodings = face_recognition.face_encodings(img_np)

    if len(face_encodings) != 1:
        raise EncodingError("The image must contain only one face")

    face_locations = face_recognition.face_locations(
        img_np, number_of_times_to_upsample=0, model="cnn"
    )

    face_images = []
    for (top, right, bottom, left) in face_locations:
        face_image = Image.fromarray(img_np[top:bottom, left:right])

        w, h = face_image.size
        # Use LANCZOS resampling here as well
        face_image = face_image.resize((HALF_SIZE, int(HALF_SIZE * (h / w))), Image.LANCZOS)

        face_images.append(face_image)

    return (face_encodings[0], face_locations[0], face_images[0])

def match_images():
    # Prompt the user for image file paths
    img1_path = input("Enter the file path for the first image: ")
    img2_path = input("Enter the file path for the second image: ")

    # Load image data
    with open(img1_path, "rb") as img1_file, open(img2_path, "rb") as img2_file:
        img1_data = img1_file.read()
        img2_data = img2_file.read()

    try:
        face_encoding1, face_location1, face_img1 = features(img1_data)
    except EncodingError as err:
        raise EncodingError("The first image must contain only one face")

    try:
        face_encoding2, face_location2, face_img2 = features(img2_data)
    except EncodingError as err:
        raise EncodingError("The second image must contain only one face")

    distance = face_recognition.face_distance([face_encoding1], face_encoding2)[0]
    matching_percentage = (1 - distance) * 100

    result = distance <= 0.6  # You can adjust the threshold as needed

    out = get_concat_h_blank(face_img1, face_img2)
    draw = ImageDraw.Draw(out)

    color = GREEN if result else RED  # Define color here

    tw, th = draw.textsize(f"Distance: {distance:.2f}", font=None)
    ow, oh = out.size

    draw.text((ow - tw, oh - th), f"Distance: {distance:.2f}", color)
    draw.text((10, 10), f"Matching Percentage: {matching_percentage:.2f}%", color)

    bio = io.BytesIO()
    out.save(bio, "PNG")

    # Save the comparison image to a file
    comparison_image_path = "comparison.png"
    with open(comparison_image_path, "wb") as comparison_file:
        comparison_file.write(bio.getbuffer())

    print("Result:", "Match" if result else "Not a Match")
    print("Distance:", distance)
    print("Matching Percentage:", matching_percentage, "%")

    # Open the saved image with the default image viewer
    out.show()

if __name__ == "__main__":
    try:
        # Filter out DeprecationWarnings
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        match_images()
    except EncodingError as err:
        print("Encoding Error:", err)