import os
import glob
import cv2
import pandas as pd
import numpy as np
from face_detection import DNNFaceDetector, _center_crop_fallback

# --- Configuration ---
DATASET_FOLDER = "datasets/train/"  # Folder containing your .npy files
OUTPUT_CSV = "annotations.csv"  # Where to save your labels
OUTPUT_FACES_DIR = "extracted_faces"  # NEW: Folder to save the cropped face images
FACE_SIZE = (150, 150)  # Size of the extracted face crops

# --- CSV Configuration ---
INFO_CSV = "datasets/train_set.csv"  # Change to your actual CSV path
CSV_FILENAME_COL = "img"  # Updated to match your 'img' column
CSV_NAME_COL = "name"  # Matches your 'name' column
CSV_CLASS_COL = "class"  # Matches your 'class' column

# --- Annotation Controls ---
LABEL_MAP = {
    "0": "Michael_Cera",
    "1": "Jesse_Eisenberg",
    "2": "Mila_Kunis",
    "3": "Sarah_Hyland",
    "4": "Not_A_Face",
}


def load_images_with_csv_info(folder_path, csv_path):
    """Loads .npy images using the exact logic from the training script."""
    print(f"Loading info from '{csv_path}'...")
    try:
        # Load the CSV exactly as your training script does
        df = pd.read_csv(csv_path, index_col=0)
        df.index = df.index.rename("id")
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return pd.DataFrame()

    print(f"Loading .npy files from '{folder_path}'...")

    images = []
    filepaths = []
    filenames = []
    valid_indices = []

    # Iterate through the CSV using the index to find the matching file
    for i, row in df.iterrows():
        filename = f"train_{i}.npy"
        filepath = os.path.join(folder_path, filename)

        if os.path.exists(filepath):
            try:
                # Load and convert BGR to RGB exactly like your original code
                img_bgr = np.load(filepath, allow_pickle=False)
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

                images.append(img_rgb)
                filepaths.append(filepath)
                filenames.append(filename)
                valid_indices.append(i)
            except Exception as e:
                print(f"Warning: Could not read {filepath}. Error: {e}")
        else:
            print(f"Warning: File not found on disk - {filepath}")

    # Filter the dataframe to only keep rows where we successfully found the image
    df = df.loc[valid_indices].copy()

    # Add the required columns for the face detector and our annotation loop
    df["img"] = images
    df["filepath"] = filepaths
    df["filename"] = filenames
    df["orig_name"] = df["name"]
    df["orig_class"] = df["class"]

    return df


def main():
    # Create the output directory for the face crops if it doesn't exist
    os.makedirs(OUTPUT_FACES_DIR, exist_ok=True)

    # 1. Load the dataset and merge with CSV
    df = load_images_with_csv_info(DATASET_FOLDER, INFO_CSV)
    if df.empty:
        print("No data loaded! Check your folder paths and CSV column names.")
        return

    print(f"Loaded {len(df)} .npy images with CSV data.")

    # 2. Initialize the detector
    print("Initializing detector...")
    detector = DNNFaceDetector(path="models", face_size=FACE_SIZE)

    # 3. Annotation Loop
    annotations = []
    print("\n--- Starting Annotation ---")
    print(f"Press the following keys to label:")
    for key, label in LABEL_MAP.items():
        print(f"  '{key}' -> {label}")
    print("Press 'q' to quit early and save progress.\n")

    # Iterate through each image in the dataframe
    for i, row in df.iterrows():
        img = row["img"]
        orig_name = row["orig_name"]
        orig_class = row["orig_class"]
        filename = row["filename"]
        base_filename = filename.replace(".npy", "")  # e.g., "train_46"

        # Get ALL faces in the image
        faces = detector.detect_faces(img)

        crops_to_annotate = []
        if not faces:
            fallback_crop = _center_crop_fallback(img, FACE_SIZE)
            crops_to_annotate.append(("Fallback", fallback_crop))
        else:
            for x, y, w, h, conf, sharp in faces:
                crop = img[y : y + h, x : x + w]
                crop_resized = cv2.resize(crop, FACE_SIZE, interpolation=cv2.INTER_AREA)
                crops_to_annotate.append(("Detected", crop_resized))

        for face_idx, (face_type, face_array) in enumerate(crops_to_annotate):

            # Scale to 0-255 if the array is normalized
            if face_array.max() <= 1.0:
                face_array = face_array * 255.0

            # This is the CLEAN, unscaled 150x150 RGB crop
            face_uint8 = face_array.astype(np.uint8)

            # Convert to BGR for OpenCV display AND saving
            clean_bgr_crop = cv2.cvtColor(face_uint8, cv2.COLOR_RGB2BGR)

            # --- Prepare the DISPLAY image (scaled up with text) ---
            display_img = cv2.resize(
                clean_bgr_crop, (400, 400), interpolation=cv2.INTER_NEAREST
            )

            cv2.putText(
                display_img,
                f"Img {i+1}/{len(df)} - Face {face_idx+1}/{len(crops_to_annotate)}",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

            csv_text = f"CSV: {orig_name} (Class {orig_class})"
            cv2.putText(
                display_img,
                csv_text,
                (10, 55),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2,
            )

            if face_type == "Fallback":
                cv2.putText(
                    display_img,
                    "NO FACE DETECTED (Fallback)",
                    (10, 380),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2,
                )

            cv2.imshow("Annotator", display_img)

            # --- Wait for input and SAVE ---
            valid_key_pressed = False
            while not valid_key_pressed:
                key = cv2.waitKey(0) & 0xFF
                char_key = chr(key)

                if char_key == "q":
                    print("Quitting early...")
                    cv2.destroyAllWindows()
                    save_annotations(annotations)
                    return

                if char_key in LABEL_MAP:
                    label = LABEL_MAP[char_key]

                    # 1. Generate a unique filename for this specific face crop
                    crop_filename = f"{base_filename}_face_{face_idx}.jpg"
                    crop_filepath = os.path.join(OUTPUT_FACES_DIR, crop_filename)

                    # 2. Save the CLEAN image to the hard drive
                    cv2.imwrite(crop_filepath, clean_bgr_crop)

                    # 3. Add the saved path to our CSV data
                    annotations.append(
                        {
                            "original_file": filename,
                            "face_index": face_idx,
                            "saved_crop_path": crop_filepath,  # NEW: Tells you exactly where the image is
                            "csv_name": orig_name,
                            "csv_class": orig_class,
                            "manual_label": label,
                        }
                    )
                    print(f"Labeled and saved: {crop_filename} -> {label}")
                    valid_key_pressed = True

    cv2.destroyAllWindows()
    save_annotations(annotations)


def save_annotations(annotations):
    if not annotations:
        print("No annotations made. Nothing to save.")
        return

    out_df = pd.DataFrame(annotations)
    out_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved {len(annotations)} annotations to {OUTPUT_CSV}")
    print(
        f"Saved {len(annotations)} cropped images to the '{OUTPUT_FACES_DIR}/' folder."
    )


if __name__ == "__main__":
    main()
