import os
import shutil
import zipfile
import random
from collections import defaultdict
from pyspark.sql import SparkSession

classes = "012345"
small_size = 200


def download_dataset():
    """Download and extract Kaggle dataset"""
    url = "https://www.kaggle.com/api/v1/datasets/download/koryakinp/fingers"

    print("Downloading dataset...")
    cmd = f'wget -c -O /tmp/data/dataset.zip "{url}"'
    result = os.system(cmd)
    if result != 0:
        raise Exception("Download failed")

    with zipfile.ZipFile("/tmp/data/dataset.zip", "r") as zip_ref:
        zip_ref.extractall("/tmp/data/fingers_data")

    return "/tmp/data/fingers_data/fingers"


def organize_images_by_class(image_files, output_base_path, dataset_name):
    """Organize images into train/val folders by class"""

    # Group files by label for stratified split
    label_groups = defaultdict(list)
    for image_path, label in image_files:
        label_groups[label].append(image_path)

    # Manual stratified split (80/20)
    random.seed(42)
    train_files = []
    val_files = []

    for label, paths in label_groups.items():
        random.shuffle(paths)
        split_idx = int(len(paths) * 0.8)

        for path in paths[:split_idx]:
            train_files.append((path, label))
        for path in paths[split_idx:]:
            val_files.append((path, label))

    for split_name, files in [("train", train_files), ("val", val_files)]:
        for image_path, label in files:
            # Create target directory structure
            target_dir = os.path.join(output_base_path, dataset_name, split_name, label)
            os.makedirs(target_dir, exist_ok=True)

            # Copy image to target location
            filename = os.path.basename(image_path)
            target_path = os.path.join(target_dir, filename)
            shutil.copy2(image_path, target_path)


def process_images(spark, data_path, output_path):
    """Process images and create dataset variants"""

    # List all image files
    image_files = []
    for root, dirs, files in os.walk(data_path):
        for file in files:
            if file.lower().endswith((".png", ".jpg", ".jpeg")):
                full_path = os.path.join(root, file)
                # Extract label from folder name
                label = file.split(".")[0].split("_")[-1][0]
                if label not in classes:
                    print("Invalid class found for sample " + full_path)
                    continue

                image_files.append((full_path, label))

    print(f"Total images found: {len(image_files)}")

    # Create small dataset
    small_files = image_files[:small_size]

    # Organize images into proper folder structure
    organize_images_by_class(small_files, output_path, "small")
    organize_images_by_class(image_files, output_path, "large")

    print(f"Small dataset: {len(small_files)} samples")
    print(f"Large dataset: {len(image_files)} samples")


def main(output_path="/tmp/data"):
    spark = SparkSession.builder.appName("FingerDataProcessor").getOrCreate()

    try:
        # Download dataset
        data_path = download_dataset()

        # Process and save
        os.makedirs(output_path, exist_ok=True)

        process_images(spark, data_path, output_path)

        print("Data processing completed successfully")

    except Exception as e:
        print(f"Error: {e}")
        raise
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
