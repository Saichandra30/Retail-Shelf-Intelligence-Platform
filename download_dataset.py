import kagglehub
import shutil
import os

print("Starting download of the retail-shelf-image dataset...")

# Download the raw dataset files
path = kagglehub.dataset_download("ankitbajaj131/retail-shelf-image")
print("Download complete!")

target_dir = os.path.abspath("datasets/retail-shelf-image")
if not os.path.exists(target_dir):
    print(f"Moving dataset to {target_dir}...")
    shutil.copytree(path, target_dir)
else:
    print(f"Dataset already exists at {target_dir}")

print("Done! Your YOLO dataset is ready at:", target_dir)
