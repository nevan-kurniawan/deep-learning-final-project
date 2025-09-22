import pathlib
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
import tarfile

# constants
ROOT_DIR = pathlib.Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Raw data paths
IMAGES_TAR_PATH = RAW_DATA_DIR / "images.tar.gz"
RAW_IMAGES_DIR = RAW_DATA_DIR / "images"
ANNOTATIONS_TAR_PATH = RAW_DATA_DIR / "annotations.tar.gz"
RAW_ANNOTATIONS_DIR = RAW_DATA_DIR / "annotations"
ANNOTATIONS_LIST_PATH = RAW_ANNOTATIONS_DIR / "list.txt"

# Flag to indicate completion of extraction
FINISH_FLAG = RAW_DATA_DIR / ".finish_flag"

# Processed data paths (output)
TRAIN_SPLIT_PATH = PROCESSED_DATA_DIR / "train.csv"
VAL_SPLIT_PATH = PROCESSED_DATA_DIR / "val.csv"
TEST_SPLIT_PATH = PROCESSED_DATA_DIR / "test.csv"


def extract_data():
    """Extracts images and annotations from .tar.gz files."""
    if FINISH_FLAG.exists():
        print("Data already extracted.")
        return

    print("--- Extracting Data ---")
    if not IMAGES_TAR_PATH.exists() or not ANNOTATIONS_TAR_PATH.exists():
        print(f"Error: Please download and place 'images.tar.gz' and 'annotations.tar.gz' in {RAW_DATA_DIR}")
        sys.exit(1)

    print(f"Extracting {IMAGES_TAR_PATH}...")
    with tarfile.open(IMAGES_TAR_PATH, "r:gz") as tar:
        tar.extractall(path=RAW_DATA_DIR)

    print(f"Extracting {ANNOTATIONS_TAR_PATH}...")
    with tarfile.open(ANNOTATIONS_TAR_PATH, "r:gz") as tar:
        tar.extractall(path=RAW_DATA_DIR)

    FINISH_FLAG.touch()
    print("Extraction complete.")


def create_splits():
    """
    Creates and validates train, validation, and test splits from the annotation list file.
    """
    print("\n--- Creating train/val/test splits ---")
    if TRAIN_SPLIT_PATH.exists() and VAL_SPLIT_PATH.exists() and TEST_SPLIT_PATH.exists():
        print("Splits already exist. Skipping creation.")
        return

    if not ANNOTATIONS_LIST_PATH.exists():
        print(f"Error: Annotation file not found at '{ANNOTATIONS_LIST_PATH}'. Please extract data first.")
        sys.exit(1)

    # --- 1. Load and Process Manifest ---
    manifest_df = pd.read_csv(
        ANNOTATIONS_LIST_PATH,
        comment='#',
        sep=' ',
        header=None,
        names=['image_id', 'class_id', 'species_id', 'breed_id']
    )
    manifest_df['label'] = manifest_df['image_id'].apply(lambda x: '_'.join(x.split('_')[:-1]))
    manifest_df['filepath'] = manifest_df['image_id'].apply(
        lambda x: (RAW_IMAGES_DIR / f"{x}.jpg").relative_to(ROOT_DIR).as_posix()
    )
    data_df = manifest_df[['filepath', 'label', 'species_id']].copy()

    # --- 2. Validation and Verification ---
    print("Validating manifest and image files...")

    # a) Verify that all image files listed in the manifest actually exist.
    missing_files_mask = data_df['filepath'].apply(lambda p: not (ROOT_DIR / p).exists())
    if missing_files_mask.any():
        num_missing = missing_files_mask.sum()
        print(f"Error: Found {num_missing} missing image files listed in the manifest.")
        print(f"Example missing file path: {ROOT_DIR / data_df[missing_files_mask]['filepath'].iloc[0]}")
        sys.exit(1)

    # b) Verify label consistency using the dataset's capitalization convention.
    # Cats (species_id=1) should have uppercase labels; Dogs (species_id=2) should have lowercase.
    is_cat = data_df['species_id'] == 1
    is_dog = data_df['species_id'] == 2
    label_is_upper = data_df['label'].str[0].str.isupper()
    label_is_lower = data_df['label'].str[0].str.islower()

    # Find entries that violate the convention.
    inconsistent_labels = data_df[~( (is_cat & label_is_upper) | (is_dog & label_is_lower) )]
    if not inconsistent_labels.empty:
        print(f"Error: Found {len(inconsistent_labels)} records with inconsistent labels in the manifest.")
        print("The label's capitalization does not match its species_id (Cat=Uppercase, Dog=Lowercase).")
        print("Example inconsistent records:")
        print(inconsistent_labels.head())
        sys.exit(1)

    print("Validation successful. All files exist and labels are consistent.")

    # --- 3. Stratified Splitting ---
    # Drop the temporary 'species_id' column before splitting and saving.
    final_df = data_df[['filepath', 'label']]
    print("Splitting data into 70% train, 15% validation, 15% test...")
    train_df, temp_df = train_test_split(
        final_df, test_size=0.3, random_state=42, stratify=final_df['label']
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, random_state=42, stratify=temp_df['label']
    )

    # --- 4. Save Manifests ---
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(TRAIN_SPLIT_PATH, index=False)
    val_df.to_csv(VAL_SPLIT_PATH, index=False)
    test_df.to_csv(TEST_SPLIT_PATH, index=False)

    print(f"Splits created successfully.")
    print(f"  - Train set:      {len(train_df)} samples")
    print(f"  - Validation set: {len(val_df)} samples")
    print(f"  - Test set:       {len(test_df)} samples")
    print(f"Manifests saved to '{PROCESSED_DATA_DIR}'")


if __name__ == "__main__":
    extract_data()
    create_splits()