import os
import pathlib
import sys
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT_DIR = pathlib.Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
IMAGES_TAR_PATH = RAW_DATA_DIR / "images.tar.gz"
RAW_IMAGES_DIR = RAW_DATA_DIR / "images"
ANNOTATIONS_TAR_PATH = RAW_DATA_DIR / "annotations.tar.gz"
RAW_ANNOTATIONS_DIR = RAW_DATA_DIR / "annotations"
FINISH_FLAG = RAW_DATA_DIR / ".finish_flag"

TRAIN_SPLIT_PATH = PROCESSED_DATA_DIR / "train.csv"
VAL_SPLIT_PATH = PROCESSED_DATA_DIR / "val.csv"
TEST_SPLIT_PATH = PROCESSED_DATA_DIR / "test.csv"

def extract_data():
    if FINISH_FLAG.exists():
        print("Data already extracted.")
        return

    print("Extracting data")

    if not IMAGES_TAR_PATH.exists() or not ANNOTATIONS_TAR_PATH.exists():
        print(f"Error. Please download and place the images.tar.gz and annotations.tar.gz in {RAW_DATA_DIR}")
        sys.exit(1)

    os.system(f"tar -xzvf {IMAGES_TAR_PATH} -C {RAW_DATA_DIR}")
    os.system(f"tar -xzvf {ANNOTATIONS_TAR_PATH} -C {RAW_DATA_DIR}")

    FINISH_FLAG.touch()

def create_splits():
    print("Creating train/test/val split")

    if TRAIN_SPLIT_PATH.exists() and VAL_SPLIT_PATH.exists() and TEST_SPLIT_PATH.exists():
        print("Train/test/val split already created.")
        return

    image_files = sorted(list(RAW_IMAGES_DIR.glob("*.jpg")))

    if not image_files:
        print("No image files found.")
        return

    labels = ["_".join(path.name.split('_')[:-1]) for path in image_files]
    data_df = pd.DataFrame({"filepath": image_files, "label": labels})

    train_df, temp_df = train_test_split(
        data_df,
        test_size=0.2, # Reserve 20% for val and test
        random_state=42,
        stratify=data_df['label']
    )

    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5, # 50% of the 20% is 10% of the total
        random_state=42,
        stratify=temp_df['label']
    )

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(TRAIN_SPLIT_PATH, index=False)
    val_df.to_csv(VAL_SPLIT_PATH, index=False)
    test_df.to_csv(TEST_SPLIT_PATH, index=False)

    print(f"Splits created. Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    print(f"Saved to {PROCESSED_DATA_DIR}")

if __name__ == "__main__":
    extract_data()
    create_splits()