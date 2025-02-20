import os
from PIL import Image

input_folder = r"camera_calibration\images\merged"
left_folder = input_folder + r"\left"
right_folder = input_folder + r"\right"

os.makedirs(left_folder, exist_ok=True)
os.makedirs(right_folder, exist_ok=True)


def split_images(input_folder, left_folder, right_folder):
    for filename in os.listdir(input_folder):
        if filename.endswith((".png", ".jpg", ".jpeg")):
            img_path = os.path.join(input_folder, filename)
            img = Image.open(img_path)
            width, height = img.size

            left_img = img.crop((0, 0, width // 2, height))
            right_img = img.crop((width // 2, 0, width, height))

            left_img.save(os.path.join(left_folder, filename))
            right_img.save(os.path.join(right_folder, filename))


split_images(input_folder, left_folder, right_folder)
