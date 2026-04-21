# %%
import os, math, shutil, threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC, LinearSVC
from skimage.feature import hog
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize
from sklearn.metrics import accuracy_score
from facenet_pytorch import InceptionResnetV1
import joblib
from tqdm import tqdm
from face_detection import DNNFaceDetector

# Local modules
from preprocessing   import (face_to_gray, convert_crops_to_gray, extract_hog_batch,
                              flatten_images, normalize_per_image, apply_pca,
                              get_train_val_split, summarize_dataset, save_crops)
from visualize       import (plot_class_distribution, plot_class_grid, plot_face_debug,
                              plot_image_sequence, plot_hog_features, plot_tsne,
                              plot_eigenfaces, plot_explained_variance,
                              plot_pca_reconstruction, plot_pca_scatter,
                              plot_validation_images, plot_confusion_matrix)
from annotate import FaceAnnotator, load_data, run_multi_annotation
import random
from urllib import request
from scipy.spatial.distance import cosine, euclidean


# %%
# Global constants used throughout the notebook
DATASET_FOLDER = "datasets/train/"  # Where the original .npy files are
ANNOTATIONS_CSV = "annotations.csv"
FACE_SIZE   = (160, 160)
TARGET_SIZE = (64, 64)     # HOG/PCA input size
CLASS_NAMES = {
    0: "Michael_Cera",
    1: "Jesse_Eisenberg",
    2: "Mila_Kunis",
    3: "Sarah_Hyland",
    4: "Not_A_Face"
}
CLASS_COLOURS = {
    0: "blue",     
    1: "orange",  
    2: "green",  
    3: "red",      
    4: "purple"    
}

os.makedirs("final_train_extracted_faces", exist_ok=True)
os.makedirs('final_data', exist_ok=True)
face_detector = DNNFaceDetector(path="models", face_size=TARGET_SIZE, confidence_threshold=0.3)

# %%
train = pd.read_csv('datasets/train_set.csv', index_col=0)
train.index = train.index.rename('id')

test = pd.read_csv('datasets/test_set.csv', index_col=0)
test.index = test.index.rename('id')

# Load raw images as RGB numpy arrays
train['img'] = [cv2.cvtColor(np.load(f'datasets/train/train_{i}.npy', allow_pickle=False),
                              cv2.COLOR_BGR2RGB) for i, _ in train.iterrows()]
test['img']  = [cv2.cvtColor(np.load(f'datasets/test/test_{i}.npy',  allow_pickle=False),
                              cv2.COLOR_BGR2RGB) for i, _ in test.iterrows()]

print(f"Training set : {len(train)} images")
print(f"Test set     : {len(test)}  images")

# %%
label_to_int = {v: k for k, v in CLASS_NAMES.items()}

# --- 2. Load the Data ---
df = pd.read_csv(ANNOTATIONS_CSV)

original_images = []
extracted_faces = []
labels = []

for _, row in df.iterrows():
    orig_file = row['original_file']     
    crop_path = row['saved_crop_path']
    label_str = row['manual_label']

    orig_path = os.path.join(DATASET_FOLDER, orig_file)

    # Only load if both the original file and your new crop exist
    if os.path.exists(orig_path) and pd.notna(crop_path) and os.path.exists(crop_path):
        
        # Load Original Image (.npy is BGR, convert to RGB)
        img_bgr = np.load(orig_path, allow_pickle=False)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        original_images.append(img_rgb)

        # Load Extracted Crop (.jpg is BGR, convert to RGB)
        crop_bgr = cv2.imread(crop_path)
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        extracted_faces.append(crop_rgb)

        # Convert label back to integer (e.g., "Mila_Kunis" -> 2)
        labels.append(label_to_int[label_str])

# --- 3. Format Variables for visualize.py ---
# plot_face_debug expects a DataFrame with an 'img' column for the originals
train = pd.DataFrame({'img': original_images})
X_train_final = np.array(extracted_faces)
y_train_final = np.array(labels)

print(f"Successfully loaded {len(X_train_final)} annotated samples.")

# %%
test_X = test['img']
test_images = []
for img in test_X:
    # Convert individual element to a numpy array
    img_arr = np.array(img, dtype=np.uint8)
    
    # Safety check: If any image is grayscale (2 dimensions instead of 3), convert it to RGB
    if len(img_arr.shape) == 2:
        img_arr = cv2.cvtColor(img_arr, cv2.COLOR_GRAY2RGB)
        
    # 3. Resize the image to exactly 160x160
    resized_img = cv2.resize(img_arr, FACE_SIZE)
    
    # Add to our new, clean list
    test_images.append(resized_img)
test_X_final = np.array(test_images)

# %%
def _to_display(img: np.ndarray) -> np.ndarray:
    """
    Converts any face-crop array to a uint8 RGB image suitable for imshow.

    Handles:
      - float32 in [0, 255]  → divide by 255, clamp, uint8
      - float32 in [0, 1]    → clamp, uint8
      - uint8                → pass through
    """
    if img.dtype == np.uint8:
        return img
    if img.max() > 1.5:  # [0, 255] float range
        return np.clip(img / 255.0, 0, 1)
    return np.clip(img, 0, 1)  # already [0, 1]


def _is_nan_face(img: np.ndarray) -> bool:
    """Returns True for sentinel NaN arrays produced by failed detection."""
    return np.isnan(img).any()

def plot_face_debug(
    data,
    face_array: np.ndarray,
    label_array: np.ndarray = None,
    class_id: int = None,
    n: int = 10,
    class_names: dict = None,
    pairs_per_row: int = 3,  # NEW: Controls how many pairs are shown horizontally
) -> None:
    """
    Shows each original image next to its extracted face crop.
    Works for both labeled training data and unlabeled test data.
    """
    # 1. Determine which indices to plot
    if label_array is not None and class_id is not None:
        # Labeled mode: filter by class
        # (Assuming _resolve_class_names handles None appropriately, fallback used just in case)
        if class_names is None:
            class_names = {}
        indices = np.where(label_array == class_id)[0]
        suptitle = f"Class {class_id} — {class_names.get(class_id, class_id)}"
    else:
        # Unlabeled/Test mode: just take the first n samples
        indices = np.arange(len(face_array))
        suptitle = "Unlabeled / Test Data Samples"

    # 2. Limit n to available data
    n = min(n, len(indices))
    if n == 0:
        print("No samples found to plot.")
        return

    # 3. Create grid layout
    num_rows = math.ceil(n / pairs_per_row)
    actual_cols = min(n, pairs_per_row) * 2  # 2 columns per pair (Original + Crop)

    # Adjust figsize to keep images square but tile them horizontally
    fig, axes = plt.subplots(
        num_rows, actual_cols, figsize=(2.5 * actual_cols, 3 * num_rows)
    )

    # Flatten axes array for easier linear iteration
    if isinstance(axes, np.ndarray):
        axes = axes.flatten()
    else:
        axes = [axes]

    fig.suptitle(f"{suptitle} ({n} samples)", fontsize=14, y=1.02)

    # Hide all axes initially (useful for empty subplots in the last row)
    for ax in axes:
        ax.axis("off")

    # 4. Plot images
    for i, idx in enumerate(indices[:n]):
        # Calculate the 1D index for the left (original) and right (face) subplots
        orig_idx = 2 * i
        face_idx = 2 * i + 1

        # Left: original image from the DataFrame
        orig = data.iloc[idx]["img"]
        axes[orig_idx].imshow(_to_display(orig.astype(np.float32)))
        axes[orig_idx].set_title(f"Orig #{idx}", fontsize=9)

        # Right: extracted face crop from the array
        face = face_array[idx]
        if _is_nan_face(face):
            axes[face_idx].text(
                0.5, 0.5, "No face", ha="center", va="center", color="red"
            )
            axes[face_idx].set_title("Detection Failed", fontsize=9, color="red")
        else:
            axes[face_idx].imshow(_to_display(face))
            axes[face_idx].set_title("Extracted", fontsize=9, color="green")

    plt.tight_layout()
    plt.show()

# %%
# for class_id in [0, 1, 2, 3, 4]:
#     n = 3 # int((y_train_final == class_id).sum())
    
#     print(f"\nPlotting {n} samples for class {class_id} ({CLASS_NAMES[class_id]})...")
#     plot_face_debug(train, X_train_final, y_train_final,
#                         class_id=class_id, n=n, class_names=CLASS_NAMES)

# %%
def augment(images, labels, num_variants=2):
    """
    Creates exactly (1 + num_variants) images per original image.
    Uses randomized combinations of mirroring, lighting, geometric shifts,
    and Cutout (Random Erasing) so the dataset doesn't explode in size.
    """
    aug_X, aug_y = [], []
    
    # Ensure images is not empty before grabbing shape
    if len(images) == 0:
        return np.array([]), np.array([])
        
    h, w = images[0].shape[:2]
    center = (w // 2, h // 2)

    # Setup CLAHE
    clahe_obj = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    for img, label in zip(images, labels):
        # 1. Always keep the clean original
        aug_X.append(img)
        aug_y.append(label)

        # 2. Generate exactly 'num_variants' additional augmented copies
        for _ in range(num_variants):
            aug_img = img.copy()

            # --- A. Random Mirror (50% chance) ---
            if random.choice([True, False]):
                aug_img = cv2.flip(aug_img, 1)

            # --- B. Random Lighting (Choose one: CLAHE, Dark, Light, or None) ---
            light_choice = random.choice(["clahe", "dark", "light", "none"])
            
            aug_img_uint8 = np.clip(aug_img, 0, 255).astype(np.uint8)
            
            if light_choice == "clahe":
                if len(aug_img_uint8.shape) == 3:
                    lab = cv2.cvtColor(aug_img_uint8, cv2.COLOR_RGB2LAB)
                    l, a, b = cv2.split(lab)
                    cl = clahe_obj.apply(l)
                    limg = cv2.merge((cl, a, b))
                    aug_img = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
                else:
                    aug_img = clahe_obj.apply(aug_img_uint8)
                    
            elif light_choice in ["dark", "light"]:
                gamma = 0.7 if light_choice == "dark" else 1.3
                invGamma = 1.0 / gamma
                table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
                aug_img = cv2.LUT(aug_img_uint8, table)

            # --- C. Random Geometric (Rotate, Scale, or None) ---
            geo_choice = random.choice(["rotate", "scale", "none"])
            
            if geo_choice == "rotate":
                angle = random.choice([-8, -4, 4, 8])
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                # BORDER_REPLICATE prevents black edges that ruin gradients
                aug_img = cv2.warpAffine(aug_img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
                
            elif geo_choice == "scale":
                scale = random.choice([0.9, 1.1])
                M = cv2.getRotationMatrix2D(center, 0, scale)
                aug_img = cv2.warpAffine(aug_img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

            # --- D. Random Cutout / Erasing (50% chance) ---
            if random.choice([True, False]):
                # Choose random box dimensions (between 15% and 35% of the image size)
                box_w = random.randint(int(w * 0.15), int(w * 0.35))
                box_h = random.randint(int(h * 0.15), int(h * 0.35))
                
                # Pick a random top-left corner for the box
                x1 = random.randint(0, w - box_w)
                y1 = random.randint(0, h - box_h)
                
                # Randomly decide between a black box or random noise
                if random.choice(["black", "noise"]) == "black":
                    # Black box (set pixels to 0)
                    aug_img[y1:y1+box_h, x1:x1+box_w] = 0
                else:
                    # Random static noise
                    if len(aug_img.shape) == 3:
                        noise = np.random.randint(0, 256, (box_h, box_w, 3), dtype=np.uint8)
                    else:
                        noise = np.random.randint(0, 256, (box_h, box_w), dtype=np.uint8)
                    aug_img[y1:y1+box_h, x1:x1+box_w] = noise
            if random.random() < 0.40:
                # Randomly pick a mild, medium, or heavy blur
                k_size = random.choice([3, 5, 7, 9])
                aug_img = cv2.GaussianBlur(aug_img, (k_size, k_size), 0)

            # Append the newly crafted variant
            aug_X.append(aug_img)
            aug_y.append(label)

    return np.array(aug_X), np.array(aug_y)

# %%
X_train_final, y_train_final = augment(X_train_final, y_train_final)

# %%
# Quick dataset summary — confirms shapes and flags any remaining NaN crops
summarize_dataset(X_train_final, y_train_final, class_names=CLASS_NAMES)

# %%
from facenet_pytorch import InceptionResnetV1
from torchvision import transforms
import torch

# %%
resnet = InceptionResnetV1(pretrained='casia-webface').eval()

# Move to GPU if available for faster extraction
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
resnet = resnet.to(device)

# %%
preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# %%
def extract_and_visualize_faces(img_array, face_detector):
    """
    Visualizes all detected faces and RETURNS the cropped, resized 
    arrays exactly as they are prepared for FaceNet.
    
    Returns:
    - A list of numpy arrays (each representing a cropped, resized face)
    """
    if not isinstance(img_array, np.ndarray):
        print("Error: Please provide a valid NumPy array.")
        return []

    img_rgb = img_array.copy()
    if img_rgb.dtype == float or img_rgb.dtype == np.float32:
        img_rgb = (img_rgb * 255).astype(np.uint8)

    # 1. Run detection
    faces = face_detector.detect_faces(img_rgb) 
    
    if not faces or len(faces) == 0:
        print("No faces detected in the image.")
        plt.imshow(img_rgb)
        plt.axis('off')
        plt.show()
        return []

    # 2. Set up Plot
    num_faces = len(faces)
    fig, axes = plt.subplots(1, num_faces + 1, figsize=(4 * (num_faces + 1), 5))
    if num_faces == 0: axes = [axes]
        
    img_with_boxes = img_rgb.copy()
    for i, (x, y, w, h, conf, sharp) in enumerate(faces):
        cv2.rectangle(img_with_boxes, (x, y), (x + w, y + h), (0, 255, 0), 3)
        cv2.putText(img_with_boxes, f"#{i+1}", (x, y - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
    axes[0].imshow(img_with_boxes)
    axes[0].set_title(f"Original ({num_faces} faces)")
    axes[0].axis('off')
    
    # 3. Process, Plot, and Store exactly what goes into FaceNet
    facenet_ready_crops = []
    
    for i, (x, y, w, h, conf, sharp) in enumerate(faces):
        # Crop the face
        face_crop = img_rgb[y : y + h, x : x + w]
        
        # Resize to your detector's face_size (e.g., 160x160)
        face_resized = cv2.resize(face_crop, face_detector.face_size, interpolation=cv2.INTER_AREA)
        
        # Plot it
        axes[i + 1].imshow(face_resized)
        axes[i + 1].set_title(f"Face {i + 1} ({face_detector.face_size})")
        axes[i + 1].axis('off')
        
        # Convert to float32 exactly like your preprocess_single method does
        final_array_for_facenet = face_resized.astype(np.float32)
        facenet_ready_crops.append(final_array_for_facenet)
        
    plt.tight_layout()
    plt.show()

    # 4. Return the list of arrays
    return facenet_ready_crops

# %%
def extract_facenet_features_batch(faces_array):
    """
    Takes a numpy array of faces of shape (N, H, W, 3) in BGR format, 
    and returns an (N, 512) array of FaceNet embeddings.
    """
    processed_tensors = []
    
    # 1. Loop through the array to apply OpenCV and preprocessing steps to each image
    for face_crop_bgr in faces_array:
        face_rgb = cv2.cvtColor(face_crop_bgr, cv2.COLOR_BGR2RGB)
        face_resized = cv2.resize(face_rgb, (160, 160))
        face_tensor = preprocess(face_resized)
        processed_tensors.append(face_tensor)
        
    # 2. Stack all individual tensors into a single batch tensor
    # Shape becomes (N, 3, 160, 160)
    batch_tensor = torch.stack(processed_tensors).to(device)
    
    # 3. Pass the entire batch through the model at once
    with torch.no_grad():
        embeddings = resnet(batch_tensor)
        
    # 4. Move back to CPU and convert to numpy. 
    # We do NOT flatten here, so the output retains the shape (N, 512)
    return embeddings.cpu().numpy()

# %%
all_embeddings = extract_facenet_features_batch(X_train_final)
all_embeddings = normalize(all_embeddings, norm='l2')

# %%
MILA_CLASS  = 2
SARAH_CLASS = 3

# ── Build a dedicated Mila-vs-Sarah training set ─────────────────────────────
mila_sarah_mask = np.isin(y_train_final, [MILA_CLASS, SARAH_CLASS])
X_ms = all_embeddings[mila_sarah_mask]
y_ms = y_train_final[mila_sarah_mask]

# %%
JESSE_CLASS = 1
MICHAEL_CLASS = 0

jesse_michael_mask = np.isin(y_train_final, [JESSE_CLASS, MICHAEL_CLASS])
X_jm = all_embeddings[jesse_michael_mask]
y_jm = y_train_final[jesse_michael_mask]

# %%
clf = SVC(kernel='rbf', C=0.1, gamma='scale', probability=True, class_weight='balanced', random_state=42)
clf.fit(all_embeddings, y_train_final)

# %%
cascade_ms_clf = SVC(kernel='rbf', C=5, gamma='scale',
                   probability=True, class_weight='balanced',
                   random_state=42)
X_ms_norm = normalize(X_ms, norm='l2')
cascade_ms_clf.fit(X_ms_norm, y_ms)

# %%
cascade_js_clf = SVC(kernel='rbf', C=5, gamma='scale',
                   probability=True, class_weight='balanced',
                   random_state=42)
X_jm_norm = normalize(X_jm, norm='l2')
cascade_js_clf.fit(X_jm_norm, y_jm)

# %%
MICHAEL_CLASS = 0
JESSE_CLASS = 1
MILA_CLASS  = 2
SARAH_CLASS = 3
OTHER_CLASS = 4

# %%
def get_tta_embedding(face_crop, feature_extractor_func):
    """
    Strictly NO-TTA. We found that 1-crop generalizes better over the Kaggle testing dataset.
    """
    augmentations = [face_crop]
    
    augmentations = [aug.astype(np.uint8) if aug.dtype != np.uint8 else aug for aug in augmentations]

    embeddings = feature_extractor_func(augmentations)

    averaged_emb = np.mean(embeddings, axis=0).reshape(1, -1)
    final_emb = normalize(averaged_emb, norm='l2')[0]

    return final_emb

# %%
def predict_with_cascade(images, main_model, cascade_ms_model, cascade_js_clf_model,
                         feature_extractor_func, face_detector, max_workers=16):
                         
    MICHAEL_CLASS, JESSE_CLASS, MILA_CLASS, SARAH_CLASS, OTHER_CLASS = 0, 1, 2, 3, 4
    all_face_crops, face_to_img_idx = [], []

    # ── 1. Image Processing & Face Detection ───────────────────────────────
    def process_single_image(img_data):
        img_index, img = img_data
        
        if isinstance(img, str):
            img_bgr = cv2.imread(img)
            if img_bgr is None:
                return img_index, []
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        else:
            img_array = img.copy()
            if img_array.dtype == float or img_array.dtype == np.float32:
                img_array = (img_array * 255).astype(np.uint8)
            img_rgb = img_array
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        faces = face_detector.detect_faces(img_rgb)
        crops = []
        
        if faces:
            for (x, y, w, h, conf, sharp) in faces:
                crop = img_bgr[y:y+h, x:x+w]
                crops.append(cv2.resize(crop, face_detector.face_size, interpolation=cv2.INTER_AREA))
        return img_index, crops
    
    with ThreadPoolExecutor(max_workers=max_workers) as exe:
        for img_idx, crops in exe.map(process_single_image, enumerate(images)):
            for crop in crops:
                all_face_crops.append(crop)
                face_to_img_idx.append(img_idx)

    # ── 2. Safety Valve ────────────────────────────────────────────────────
    if not all_face_crops:
        return [OTHER_CLASS] * len(images), []

    # ── 3. Test-Time Augmentation (TTA) ────────────────────────────────────
    tta_embeddings = []
    for crop in all_face_crops:
        stable_emb = get_tta_embedding(crop, feature_extractor_func)
        tta_embeddings.append(stable_emb)
        
    embeddings = np.array(tta_embeddings)

    # ── 4. Main Model Predictions ──────────────────────────────────────────
    main_probs = main_model.predict_proba(embeddings)      

    # ── 5. Cascade Classification & Strict Logic ───────────────────────────
    face_preds = []
    
    mila_idx  = list(cascade_ms_model.classes_).index(MILA_CLASS)
    sarah_idx = list(cascade_ms_model.classes_).index(SARAH_CLASS)
    jesse_idx = list(cascade_js_clf_model.classes_).index(JESSE_CLASS)
    michael_idx = list(cascade_js_clf_model.classes_).index(MICHAEL_CLASS)


    KNOWN_THRESHOLD = 0.30 

    for i, prob in enumerate(main_probs):
        pred_cls = np.argmax(prob)
        if prob[pred_cls] >= KNOWN_THRESHOLD and pred_cls != OTHER_CLASS:
            face_preds.append((pred_cls, prob[pred_cls]))
        else:
            face_preds.append((OTHER_CLASS, prob[OTHER_CLASS]))

    # ── 6. Image Mapping & Priority Rules ──────────────────────────────────
    preds_per_image = {}
    for img_idx, pred_data in zip(face_to_img_idx, face_preds):
        preds_per_image.setdefault(img_idx, []).append(pred_data)

    final_results = [OTHER_CLASS] * len(images) 
    
    for i in range(len(images)):
        if i in preds_per_image:
            ps = preds_per_image[i] 
            
            class_max_probs = {}
            for cls, prob in ps:
                class_max_probs[cls] = max(class_max_probs.get(cls, 0), prob)
                
            if JESSE_CLASS in class_max_probs and MICHAEL_CLASS in class_max_probs:
                if class_max_probs[JESSE_CLASS] > 0.75:
                    class_max_probs[MICHAEL_CLASS] = 0.0 
                    
            valid_competitors = {k: v for k, v in class_max_probs.items() if k in (0, 1, 2, 3)}
            
            if valid_competitors:
                winner = max(valid_competitors, key=valid_competitors.get)
                final_results[i] = winner

    return final_results, main_probs

def main():
    # 1. Extract all raw images from the dataframe into a Python list
    all_raw_images = test['img'].tolist()

    final_test_predictions = []

    # Set a batch size 
    batch_size = 48

    # 2. Loop through the list in chunks
    for i in tqdm(range(0, len(all_raw_images), batch_size), desc="Classifying Test Batches"):
        
        # Slice the list to get the current batch of images
        batch_images = all_raw_images[i:i + batch_size]
        
        # Run our new batch function on this chunk
        batch_labels, probs = predict_with_cascade(
            images=batch_images, 
            main_model=clf, # Make sure to pass your actual trained model here
            cascade_ms_model=cascade_ms_clf,
            cascade_js_clf_model=cascade_js_clf,
            feature_extractor_func=extract_facenet_features_batch, 
            face_detector=face_detector
        )
        
        # Extend the final list with the results from this batch
        final_test_predictions.extend(batch_labels)

    # Attach the predictions directly back to your dataframe for easy viewing
    test['predictions'] = final_test_predictions

    # subset testing
    y_test_true = np.load('final_data/y_test.npy')[:len(final_test_predictions)]
    acc = accuracy_score(y_test_true, final_test_predictions)
    print(f"\nSubset Accuracy Score on first {len(final_test_predictions)} images: {acc:.5f}")
    from sklearn.metrics import classification_report
    print(classification_report(y_test_true, final_test_predictions))
    
    test['class'] = test['predictions']
    test.drop(columns=['img', 'predictions'], inplace=True)
    test.to_csv('submission.csv')
    print("Saved submission.csv successfully!")
    
if __name__ == '__main__':
    main()

# %%
def get_images_with_multiple_faces(images, face_detector, max_workers=4):
    """
    Takes a list of images (file paths or BGR numpy arrays), detects faces, 
    and returns a list of indices (IDs) for images containing more than one face.
    """
    multiple_faces_indices = []
    
    # --- HELPER FUNCTION FOR PARALLEL EXECUTION ---
    def process_single_image(img_data):
        img_index, img = img_data
        
        # If the input is a file path, load it
        if isinstance(img, str):
            img_bgr = cv2.imread(img)
            if img_bgr is None:
                print(f"Warning: Could not load image at {img}")
                return img_index, 0
        else:
            img_bgr = img # Assuming it's already a loaded BGR numpy array
            
        # Detect faces on the actual image
        faces = face_detector.detect_faces(img_bgr)
        
        # Return the index and the total count of faces found
        face_count = len(faces) if faces else 0
        return img_index, face_count

    # PARALLELIZE IMAGE LOADING AND FACE DETECTION
    # executor.map ensures results are returned in the original order
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = executor.map(process_single_image, enumerate(images))
        
        # Unpack the results and filter for > 1 face
        for img_idx, face_count in results:
            if face_count > 1:
                multiple_faces_indices.append(img_idx)
                
    return multiple_faces_indices

# multi_faces_indices = get_images_with_multiple_faces(all_raw_images, face_detector)
# len(multi_faces_indices)

# %%
def get_images_with_no_faces(images, face_detector, max_workers=4):
    """
    Takes a list of images (file paths or BGR numpy arrays), detects faces, 
    and returns a list of indices (IDs) for images containing no faces.
    """
    no_faces_indices = []
    
    # --- HELPER FUNCTION FOR PARALLEL EXECUTION ---
    def process_single_image(img_data):
        img_index, img = img_data
        
        # If the input is a file path, load it
        if isinstance(img, str):
            img_bgr = cv2.imread(img)
            if img_bgr is None:
                print(f"Warning: Could not load image at {img}")
                return img_index, 0
        else:
            img_bgr = img # Assuming it's already a loaded BGR numpy array
            
        # Detect faces on the actual image
        faces = face_detector.detect_faces(img_bgr)
        
        # Return the index and the total count of faces found
        face_count = len(faces) if faces else 0
        return img_index, face_count

    # PARALLELIZE IMAGE LOADING AND FACE DETECTION
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = executor.map(process_single_image, enumerate(images))
        
        # Unpack the results and filter for 0 faces
        for img_idx, face_count in results:
            if face_count == 0:
                no_faces_indices.append(img_idx)
                
    return no_faces_indices

