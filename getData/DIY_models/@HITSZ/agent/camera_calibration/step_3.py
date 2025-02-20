import os
import shutil

source_dirs = [
    r"camera_calibration\images\backup_1",
    r"camera_calibration\images\backup_2",
    r"camera_calibration\images\backup_3",
    r"camera_calibration\images\backup_4",
]
target_dir = r"camera_calibration\images\merged"

if not os.path.exists(target_dir):
    os.makedirs(target_dir)


def get_new_filename(filename, existing_files):
    name, ext = os.path.splitext(filename)
    num = int(name)
    while filename in existing_files:
        num += 1
        filename = f"{num}{ext}"
    return filename


existing_files = set(os.listdir(target_dir))

for source_dir in source_dirs:
    for filename in os.listdir(source_dir):
        if filename.endswith(".png"):
            new_filename = filename
            if new_filename in existing_files:
                new_filename = get_new_filename(new_filename, existing_files)
            existing_files.add(new_filename)
            shutil.copy(
                os.path.join(source_dir, filename),
                os.path.join(target_dir, new_filename),
            )
