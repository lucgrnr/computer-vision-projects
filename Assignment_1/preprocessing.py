"""
preprocessing.py
----------------
Data transformation utilities for KUL H02A5a Computer Vision Group Assignment 1.

Transforms raw face crops (produced by face_detection.py) into feature vectors
ready for classifiers.

Functions
---------
    face_to_gray            : DNN/HAAR crop → uint8 grayscale, resized
    convert_crops_to_gray   : apply face_to_gray to a whole array
    flatten_images          : (N, H, W) → (N, H*W) for PCA input
    normalize_per_image     : zero-mean, unit-std per image (illumination invariance)
    extract_hog_batch       : apply a HOGFeatureExtractor to a full array
    get_train_val_split     : stratified index-based split returning aligned arrays
    apply_pca               : fit or reuse a PCA and return scores + model
    safe_stack              : stack a list of arrays, replacing NaN images with zeros

Pipeline summary
----------------
    Raw images
        │
        ▼  face_detection.py
    Face crops  (N, 100, 100, 3)  float32 RGB
        │
        ▼  convert_crops_to_gray
    Gray crops  (N, 64, 64)       uint8
        │
        ├──▶  extract_hog_batch  →  HOG features  (N, 1764)
        │
        └──▶  flatten_images
              normalize_per_image
              apply_pca           →  PCA scores    (N, p)
"""

import cv2
import numpy as np
import os
from sklearn.model_selection import train_test_split


# =============================================================================
# 1. Image conversion
# =============================================================================


def face_to_gray(
    img_array: np.ndarray,
    target_size: tuple = (64, 64),
) -> np.ndarray:
    """
    Converts a single face crop to a uint8 grayscale image at *target_size*.

    Input is expected to be a float32 RGB array in [0, 255] as produced by
    DNNFaceDetector / HAARPreprocessor.  NaN crops (from failed detections)
    are silently converted to zero images so the pipeline never crashes —
    those samples simply get a zero feature vector downstream.

    Parameters
    ----------
    img_array   : face crop, shape (H, W, 3) float32, or NaN sentinel array
    target_size : (width, height) output size — (64, 64) is the standard
                  HOG input size and works well for PCA too

    Returns
    -------
    np.ndarray of shape (height, width), dtype uint8
    """
    if np.isnan(img_array).any():
        return np.zeros((target_size[1], target_size[0]), dtype=np.uint8)

    img_uint8 = np.clip(img_array, 0, 255).astype(np.uint8)

    if img_uint8.ndim == 3:
        img_gray = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)
    else:
        img_gray = img_uint8  # already grayscale

    return cv2.resize(img_gray, target_size, interpolation=cv2.INTER_AREA)


def convert_crops_to_gray(
    face_array: np.ndarray,
    target_size: tuple = (64, 64),
    verbose: bool = True,
) -> np.ndarray:
    """
    Applies face_to_gray to every crop in *face_array*.

    Parameters
    ----------
    face_array  : array of shape (N, H, W, 3), float32 RGB
    target_size : output size passed to face_to_gray
    verbose     : print progress summary if True

    Returns
    -------
    np.ndarray of shape (N, target_size[1], target_size[0]), dtype uint8
    """
    result = np.array([face_to_gray(img, target_size) for img in face_array])

    if verbose:
        nan_count = sum(np.isnan(img).any() for img in face_array)
        print("convert_crops_to_gray:")
        print(f"  Input  : {face_array.shape}")
        print(f"  Output : {result.shape}")
        if nan_count:
            print(f"  Warning: {nan_count} NaN crops replaced with zero images")

    return result


# =============================================================================
# 2. Array reshaping
# =============================================================================


def flatten_images(imgs: np.ndarray) -> np.ndarray:
    """
    Flattens a batch of images to a 2-D feature matrix.

    (N, H, W)    → (N, H*W)
    (N, H, W, C) → (N, H*W*C)

    Required before PCA since sklearn's PCA expects (n_samples, n_features).

    Parameters
    ----------
    imgs : array of shape (N, H, W) or (N, H, W, C)

    Returns
    -------
    np.ndarray of shape (N, H*W) or (N, H*W*C), dtype float32
    """
    n = imgs.shape[0]
    return imgs.reshape(n, -1).astype(np.float32)


# =============================================================================
# 3. Normalisation
# =============================================================================


def normalize_per_image(imgs_flat: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Subtracts the per-image mean and divides by the per-image standard deviation.

    WHY: Face images vary in global brightness (studio lighting vs outdoor) and
    contrast.  Per-image normalisation removes these global effects so the
    classifier focuses on the relative pixel structure (facial features) rather
    than absolute brightness.

    This is done PER IMAGE (not per pixel / per feature) because:
      - Per-pixel normalisation removes spatial structure that HOG and PCA need.
      - Global dataset normalisation doesn't handle outlier images well with
        only 80 training samples.

    Parameters
    ----------
    imgs_flat : (N, D) float32 array — each row is one flattened image
    eps       : small constant to prevent division by zero for blank images

    Returns
    -------
    np.ndarray of shape (N, D), float32, each row zero-mean unit-variance
    """
    mean = imgs_flat.mean(axis=1, keepdims=True)
    std = imgs_flat.std(axis=1, keepdims=True) + eps
    return ((imgs_flat - mean) / std).astype(np.float32)


def compute_mean_face(imgs_flat: np.ndarray) -> np.ndarray:
    """
    Returns the dataset mean face vector (mean across all N images).

    Required for PCA / eigenfaces:
      1. Compute mean face.
      2. Subtract from every image → centred data matrix.
      3. Apply SVD/PCA to the centred matrix.

    Parameters
    ----------
    imgs_flat : (N, D) float32 array

    Returns
    -------
    np.ndarray of shape (D,), float32
    """
    return imgs_flat.mean(axis=0).astype(np.float32)


# =============================================================================
# 4. HOG feature extraction
# =============================================================================


def extract_hog_batch(
    gray_imgs: np.ndarray,
    extractor,
    verbose: bool = True,
) -> np.ndarray:
    """
    Applies *extractor*.transform() to every image in *gray_imgs*.

    Designed to work with HOGFeatureExtractor from the notebook, or any
    object that implements a .transform(img) → 1-D array interface.

    NaN/zero images (failed detections) produce a zero feature vector of
    the correct length so array shapes stay consistent.

    Parameters
    ----------
    gray_imgs : (N, H, W) uint8 grayscale array
    extractor : HOGFeatureExtractor instance (or compatible object)
    verbose   : print shape summary if True

    Returns
    -------
    np.ndarray of shape (N, feature_dim), float32
    """
    features = []
    ref_len = None  # determined from first successful extraction

    for img in gray_imgs:
        if img.max() == 0:
            # Blank image — append zero vector once we know the feature length
            if ref_len is not None:
                features.append(np.zeros(ref_len, dtype=np.float32))
            else:
                features.append(None)  # resolved in the second pass below
            continue

        feat = extractor.transform(img).astype(np.float32)
        ref_len = len(feat)
        features.append(feat)

    # Second pass: replace any None placeholders (blank images before first
    # valid image) now that we know ref_len
    if ref_len is None:
        raise ValueError(
            "extract_hog_batch: all images are blank — "
            "face detection may have failed for the entire set."
        )

    features = [
        f if f is not None else np.zeros(ref_len, dtype=np.float32) for f in features
    ]
    result = np.stack(features).astype(np.float32)

    if verbose:
        print("extract_hog_batch:")
        print(f"  Input  : {gray_imgs.shape}")
        print(f"  Output : {result.shape}  (feature vector length = {ref_len})")

    return result


# =============================================================================
# 5. Train / validation split
# =============================================================================


def get_train_val_split(
    *arrays,
    labels: np.ndarray,
    test_size: float = 0.25,
    random_state: int = 42,
    verbose: bool = True,
):
    """
    Stratified train/validation split that keeps multiple arrays aligned.

    Splitting by INDEX guarantees that features, gray images, face crops,
    and any other aligned arrays are split identically — avoids the subtle
    bug where calling train_test_split twice with the same random_state
    only works if the data is identical, which breaks silently after any
    preprocessing change.

    Parameters
    ----------
    *arrays      : any number of arrays to split, all must have the same N
                   e.g. get_train_val_split(hog_features, gray_imgs, labels=y)
    labels       : 1-D integer label array used for stratification
    test_size    : fraction of data for validation (default 0.25)
    random_state : fixed seed for reproducibility
    verbose      : print split summary if True

    Returns
    -------
    tuple of (train_array_1, val_array_1, train_array_2, val_array_2, ...)
    plus (train_labels, val_labels) appended at the end

    Example
    -------
    X_train, X_val, imgs_train, imgs_val, y_train, y_val = get_train_val_split(
        hog_features, gray_imgs, labels=train_y_dnn
    )
    """
    n = len(labels)
    for i, arr in enumerate(arrays):
        if len(arr) != n:
            raise ValueError(
                f"Array {i} has length {len(arr)} but labels has length {n}. "
                "All arrays must have the same number of samples."
            )

    indices = np.arange(n)
    train_idx, val_idx = train_test_split(
        indices,
        test_size=test_size,
        random_state=random_state,
        stratify=labels,
    )

    result = []
    for arr in arrays:
        result.append(arr[train_idx])
        result.append(arr[val_idx])

    # Append labels last
    result.append(labels[train_idx])
    result.append(labels[val_idx])

    if verbose:
        print("get_train_val_split:")
        print(f"  Total samples  : {n}")
        print(f"  Training       : {len(train_idx)}")
        print(f"  Validation     : {len(val_idx)}")
        print(
            f"  Train dist     : {dict(zip(*np.unique(labels[train_idx], return_counts=True)))}"
        )
        print(
            f"  Val   dist     : {dict(zip(*np.unique(labels[val_idx],   return_counts=True)))}"
        )

    return tuple(result)


# =============================================================================
# 6. PCA
# =============================================================================


def apply_pca(
    train_flat: np.ndarray,
    val_flat: np.ndarray = None,
    test_flat: np.ndarray = None,
    n_components: int = None,
    whiten: bool = False,
    verbose: bool = True,
):
    """
    Fits PCA on *train_flat* and projects train / val / test sets.

    WHY fit only on train:
      Fitting PCA on the full dataset (including val/test) would leak
      information about those images into the feature space — the principal
      components would partially explain test-set variation rather than
      purely training-set variation.

    Parameters
    ----------
    train_flat   : (N_train, D) float32 — flattened training images
    val_flat     : (N_val,   D) float32 — optional validation images
    test_flat    : (N_test,  D) float32 — optional test images
    n_components : number of PCA components to keep.
                   None = keep all (min(N, D)) components.
                   Choose based on explained variance plot — typically the
                   number that explains 95 % of variance.
    whiten       : divide scores by sqrt(eigenvalue) to make components
                   unit-variance — can help SVM performance
    verbose      : print explained variance summary if True

    Returns
    -------
    pca          : fitted sklearn PCA object (access .components_, .mean_, etc.)
    train_scores : (N_train, n_components) projected training features
    val_scores   : (N_val,   n_components) or None
    test_scores  : (N_test,  n_components) or None
    """
    from sklearn.decomposition import PCA

    pca = PCA(n_components=n_components, whiten=whiten, random_state=42)
    train_scores = pca.fit_transform(train_flat)

    val_scores = pca.transform(val_flat) if val_flat is not None else None
    test_scores = pca.transform(test_flat) if test_flat is not None else None

    if verbose:
        cumvar = np.cumsum(pca.explained_variance_ratio_)
        idx_90 = int(np.searchsorted(cumvar, 0.90)) + 1
        idx_95 = int(np.searchsorted(cumvar, 0.95)) + 1
        idx_99 = int(np.searchsorted(cumvar, 0.99)) + 1

        print("apply_pca:")
        print(f"  Input dimension       : {train_flat.shape[1]}")
        print(f"  Components kept       : {pca.n_components_}")
        print(f"  Non-zero eigenvalues  : {(pca.explained_variance_ > 1e-10).sum()}")
        print(f"  Components for 90 %   : {idx_90}")
        print(f"  Components for 95 %   : {idx_95}")
        print(f"  Components for 99 %   : {idx_99}")
        print(f"  Train scores shape    : {train_scores.shape}")
        if val_scores is not None:
            print(f"  Val   scores shape    : {val_scores.shape}")
        if test_scores is not None:
            print(f"  Test  scores shape    : {test_scores.shape}")

    return pca, train_scores, val_scores, test_scores


# =============================================================================
# 7. Miscellaneous
# =============================================================================


def safe_stack(arrays: list, fill_value: float = 0.0) -> np.ndarray:
    """
    Stacks a list of arrays into a single ndarray, replacing any NaN arrays
    with *fill_value*.

    Useful when face_detection returns a mix of valid crops and NaN sentinels
    and you need a clean numeric array for sklearn.

    Parameters
    ----------
    arrays     : list of np.ndarray — all must have the same shape
    fill_value : value to substitute for NaN arrays (default 0.0)

    Returns
    -------
    np.ndarray with NaN arrays replaced by fill_value arrays
    """
    cleaned = []
    for arr in arrays:
        if np.isnan(arr).any():
            cleaned.append(np.full_like(arr, fill_value, dtype=np.float32))
        else:
            cleaned.append(arr.astype(np.float32))
    return np.stack(cleaned)


def summarize_dataset(
    face_array: np.ndarray,
    label_array: np.ndarray,
    class_names: dict = None,
) -> None:
    """
    Prints a concise summary of a preprocessed dataset.
    Useful at the start of each section to confirm the data looks right.

    Parameters
    ----------
    face_array  : (N, H, W, 3) or (N, H, W) array of face crops
    label_array : integer labels aligned with face_array
    class_names : dict mapping int → display string
    """
    if class_names is None:
        class_names = {0: "Michael/Sarah", 1: "Jesse", 2: "Mila"}

    nan_count = sum(np.isnan(img).any() for img in face_array)

    print("Dataset summary:")
    print(f"  Total samples : {len(face_array)}")
    print(f"  Image shape   : {face_array.shape[1:]}")
    print(f"  dtype         : {face_array.dtype}")
    print(f"  NaN crops     : {nan_count}")
    print("  Class counts  :")
    for cls, cnt in zip(*np.unique(label_array, return_counts=True)):
        print(f"    {cls} — {class_names.get(cls, cls)}: {cnt}")


def save_crops(
    face_array: np.ndarray,
    label_array: np.ndarray,
    train_df,
    output_dir: str = "extracted_faces",
    class_names: dict = None,
) -> None:
    """
    Saves all extracted face crops to disk as PNG files, organised by class.

    Directory structure
    -------------------
    extracted_faces/
        Jesse/
            0_Jesse_Eisenberg.png
            1_Jesse_Eisenberg.png
            ...
        Mila/
            0_Mila_Kunis.png
            ...
        Michael_Sarah/
            0_Michael_Cera.png
            ...
        metadata.json   ← index, original name, class, source for every crop

    Parameters
    ----------
    face_array  : (N, H, W, 3) float32 array of face crops
    label_array : integer labels aligned with face_array
    train_df    : DataFrame with 'name' column for filenames
    output_dir  : root directory to write into (created if absent)
    class_names : dict mapping int → folder name
                  defaults to {0:'Michael_Sarah', 1:'Jesse', 2:'Mila'}
    """
    import json

    if class_names is None:
        class_names = {0: "Michael_Sarah", 1: "Jesse", 2: "Mila"}

    # Create one subdirectory per class
    for name in class_names.values():
        os.makedirs(os.path.join(output_dir, name), exist_ok=True)

    metadata = []

    for idx, (img, label) in enumerate(zip(face_array, label_array)):
        class_folder = class_names.get(int(label), f"class_{label}")
        person_name = train_df.iloc[idx]["name"]

        if np.isnan(img).any():
            # Still record it in metadata but skip saving
            metadata.append(
                {
                    "index": idx,
                    "name": person_name,
                    "class": int(label),
                    "file": None,
                    "note": "no face detected — crop not saved",
                }
            )
            continue

        filename = f"{idx}_{person_name}.png"
        filepath = os.path.join(output_dir, class_folder, filename)

        # Convert float32 RGB → uint8 BGR for cv2.imwrite
        img_uint8 = np.clip(img, 0, 255).astype(np.uint8)
        img_bgr = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR)
        cv2.imwrite(filepath, img_bgr)

        metadata.append(
            {
                "index": idx,
                "name": person_name,
                "class": int(label),
                "file": os.path.join(class_folder, filename),
            }
        )

    # Save metadata so you always know which file came from which image
    meta_path = os.path.join(output_dir, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    saved = sum(1 for m in metadata if m["file"] is not None)
    print(f"save_crops: saved {saved}/{len(face_array)} crops to '{output_dir}/'")
    print(f"  Metadata written to '{meta_path}'")
