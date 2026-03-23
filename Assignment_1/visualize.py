"""
visualize.py
------------
Visualization utilities for KUL H02A5a Computer Vision Group Assignment 1.

Functions
---------
    plot_image_sequence       : grid of face crops (with NaN handling)
    plot_face_debug           : original image vs extracted crop side by side
    plot_class_grid           : one row per class, quick dataset overview
    plot_validation_images    : validation crops with true/predicted labels
    plot_hog_features         : face crop next to its HOG visualization
    plot_tsne                 : 2-D t-SNE scatter coloured by class
    plot_eigenfaces           : top-N eigenfaces as a grid
    plot_pca_reconstruction   : face reconstructed with increasing components
    plot_pca_scatter          : faces projected onto first two PCs
    plot_confusion_matrix     : styled confusion matrix
    plot_class_distribution   : bar chart of class counts

All functions follow the same conventions:
    - class_names : dict {int → str} mapping label integers to display names
                    defaults to {0: 'Michael/Sarah', 1: 'Jesse', 2: 'Mila'}
    - Images are expected as float32 RGB in [0, 255] or uint8 RGB.
    - NaN arrays (from failed face detection) are detected and labelled
      "No face" instead of crashing.
"""

import math

import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# Default class label mapping used throughout the assignment
DEFAULT_CLASS_NAMES = {0: "Michael/Sarah", 1: "Jesse", 2: "Mila"}

# Consistent per-class colours so all plots are visually linked
CLASS_COLOURS = {0: "#4C72B0", 1: "#DD8452", 2: "#55A868"}


# =============================================================================
# Internal helpers
# =============================================================================


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


def _resolve_class_names(class_names):
    if class_names is None:
        return DEFAULT_CLASS_NAMES
    if isinstance(class_names, list):
        # Convert ['Michael/Sarah', 'Jesse', 'Mila'] → {0: 'Michael/Sarah', 1: 'Jesse', 2: 'Mila'}
        return {i: name for i, name in enumerate(class_names)}
    return class_names  # already a dict


# =============================================================================
# 1. Basic grid plots
# =============================================================================


def plot_image_sequence(
    data: np.ndarray,
    n: int,
    imgs_per_row: int = 10,
    title: str = "",
) -> None:
    """
    Plots the first *n* images from *data* in a grid.

    Parameters
    ----------
    data        : array of shape (N, H, W, 3)
    n           : number of images to show
    imgs_per_row: columns in the grid
    title       : optional suptitle
    """
    n = min(n, len(data))
    n_rows = math.ceil(n / imgs_per_row)
    n_cols = min(imgs_per_row, n)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.5 * n_cols, 2.5 * n_rows))
    axes = np.array(axes).flatten() if n > 1 else [axes]

    for i, ax in enumerate(axes):
        ax.axis("off")
        if i < n:
            img = data[i]
            if _is_nan_face(img):
                ax.set_title("No face", fontsize=7)
            else:
                ax.imshow(_to_display(img))

    if title:
        fig.suptitle(title, fontsize=12, y=1.01)
    plt.tight_layout()
    plt.show()


def plot_face_debug(
    data,
    face_array: np.ndarray,
    label_array: np.ndarray,
    class_id: int,
    n: int = 10,
    class_names: dict = None,
) -> None:
    """
    Shows each original image next to its extracted face crop.
    Useful for validating that the face detector is picking the right person.

    Parameters
    ----------
    data        : DataFrame with an 'img' column (raw images)
    face_array  : preprocessed face crops, shape (N, H, W, 3)
    label_array : integer class labels aligned with face_array
    class_id    : which class to visualise
    n           : how many pairs to show
    """
    class_names = _resolve_class_names(class_names)
    indices = np.where(label_array == class_id)[0]
    n = min(n, len(indices))

    fig, axes = plt.subplots(n, 2, figsize=(6, 2.5 * n))
    # Handle edge case of a single row
    if n == 1:
        axes = axes[np.newaxis, :]

    fig.suptitle(
        f"Class {class_id} — {class_names.get(class_id, class_id)} " f"({n} samples)",
        fontsize=12,
    )

    for row, idx in enumerate(indices[:n]):
        # Left: original image
        orig = data.iloc[idx]["img"]
        axes[row, 0].imshow(_to_display(orig.astype(np.float32)))
        axes[row, 0].set_title(f"Original #{idx}", fontsize=8)
        axes[row, 0].axis("off")

        # Right: extracted face crop
        face = face_array[idx]
        if _is_nan_face(face):
            axes[row, 1].set_title("No face detected", fontsize=8, color="red")
        else:
            axes[row, 1].imshow(_to_display(face))
            axes[row, 1].set_title("Extracted face", fontsize=8, color="green")
        axes[row, 1].axis("off")

    plt.tight_layout()
    plt.show()


def plot_class_grid(
    face_array: np.ndarray,
    label_array: np.ndarray,
    imgs_per_class: int = 10,
    class_names: dict = None,
) -> None:
    """
    One row per class — gives a quick overview of the whole dataset.

    Parameters
    ----------
    face_array     : preprocessed face crops, shape (N, H, W, 3)
    label_array    : integer labels aligned with face_array
    imgs_per_class : how many samples to show per class (columns)
    """
    class_names = _resolve_class_names(class_names)
    classes = sorted(np.unique(label_array))
    n_rows = len(classes)
    n_cols = imgs_per_class

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.2 * n_cols, 2.5 * n_rows))

    for row, cls in enumerate(classes):
        indices = np.where(label_array == cls)[0]
        label = class_names.get(cls, str(cls))

        for col in range(n_cols):
            ax = axes[row, col]
            ax.axis("off")
            if col < len(indices):
                img = face_array[indices[col]]
                if not _is_nan_face(img):
                    ax.imshow(_to_display(img))
                else:
                    ax.set_title("No face", fontsize=6, color="red")
            # Row label on the leftmost cell only
            if col == 0:
                ax.set_ylabel(
                    label,
                    fontsize=10,
                    rotation=0,
                    labelpad=60,
                    va="center",
                    color=CLASS_COLOURS.get(cls, "black"),
                )

    fig.suptitle("Dataset overview — one row per class", fontsize=13)
    plt.tight_layout()
    plt.show()


# =============================================================================
# 2. Validation / prediction plots
# =============================================================================


def plot_validation_images(
    X_val_images: np.ndarray,
    y_val_true: np.ndarray,
    y_val_pred: np.ndarray = None,
    class_names: dict = None,
    imgs_per_row: int = 5,
) -> None:
    """
    Plots validation images with true labels.
    When *y_val_pred* is supplied, titles turn green (correct) or red (wrong).

    Parameters
    ----------
    X_val_images : array of images, shape (N, H, W) or (N, H, W, 3)
    y_val_true   : true integer labels
    y_val_pred   : predicted integer labels (optional)
    class_names  : dict mapping int → display string
    imgs_per_row : grid columns
    """
    class_names = _resolve_class_names(class_names)
    n = len(X_val_images)
    n_cols = min(imgs_per_row, n)
    n_rows = math.ceil(n / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3.5 * n_rows))
    axes = np.array(axes).flatten() if n > 1 else [axes]

    for i, ax in enumerate(axes):
        ax.axis("off")
        if i >= n:
            continue

        img = X_val_images[i]
        cmap = "gray" if img.ndim == 2 else None
        ax.imshow(_to_display(img) if img.ndim == 3 else img, cmap=cmap)

        true_label = class_names.get(y_val_true[i], y_val_true[i])
        title = f"True: {true_label}"
        colour = "black"

        if y_val_pred is not None:
            pred_label = class_names.get(y_val_pred[i], y_val_pred[i])
            title += f"\nPred: {pred_label}"
            colour = "green" if y_val_true[i] == y_val_pred[i] else "red"

        ax.set_title(title, color=colour, fontsize=9)

    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: dict = None,
    title: str = "Confusion Matrix",
) -> None:
    """
    Styled confusion matrix with per-cell counts and normalised percentages.
    """
    from sklearn.metrics import confusion_matrix

    class_names = _resolve_class_names(class_names)
    classes = sorted(np.unique(np.concatenate([y_true, y_pred])))
    labels = [class_names.get(c, str(c)) for c in classes]

    cm = confusion_matrix(y_true, y_pred, labels=classes)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for i in range(len(classes)):
        for j in range(len(classes)):
            colour = "white" if cm_norm[i, j] > 0.6 else "black"
            ax.text(
                j,
                i,
                f"{cm[i, j]}\n({cm_norm[i, j]*100:.0f}%)",
                ha="center",
                va="center",
                fontsize=11,
                color=colour,
            )

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted label", fontsize=11)
    ax.set_ylabel("True label", fontsize=11)
    ax.set_title(title, fontsize=13)

    plt.tight_layout()
    plt.show()


def plot_class_distribution(
    label_array: np.ndarray,
    class_names: dict = None,
    title: str = "Class distribution",
) -> None:
    """Bar chart of sample counts per class."""
    class_names = _resolve_class_names(class_names)
    classes, counts = np.unique(label_array, return_counts=True)
    labels = [class_names.get(c, str(c)) for c in classes]
    colours = [CLASS_COLOURS.get(c, "#888888") for c in classes]

    fig, ax = plt.subplots(figsize=(6, 3.5))
    bars = ax.bar(labels, counts, color=colours, edgecolor="white", linewidth=0.8)

    for bar, count in zip(bars, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.3,
            str(count),
            ha="center",
            va="bottom",
            fontsize=11,
        )

    ax.set_ylabel("Number of images")
    ax.set_title(title, fontsize=12)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.show()


# =============================================================================
# 3. Feature visualization
# =============================================================================


def plot_hog_features(
    face_array: np.ndarray,
    label_array: np.ndarray,
    hog_params: dict = None,
    n_per_class: int = 2,
    class_names: dict = None,
) -> None:
    """
    For each class shows *n_per_class* faces and their HOG visualisations
    side by side.  Helps verify that HOG is capturing meaningful facial
    structure (eye sockets, nose bridge, jawline) rather than background.

    Parameters
    ----------
    face_array  : preprocessed gray face crops, shape (N, H, W)
    label_array : integer labels
    hog_params  : dict passed to skimage.feature.hog; uses sensible defaults
    n_per_class : samples per class to display
    """
    from skimage.feature import hog

    class_names = _resolve_class_names(class_names)
    if hog_params is None:
        hog_params = dict(
            orientations=9,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            block_norm="L2-Hys",
        )

    classes = sorted(np.unique(label_array))
    n_rows = len(classes) * n_per_class
    fig, axes = plt.subplots(n_rows, 2, figsize=(5, 2.5 * n_rows))

    row = 0
    for cls in classes:
        indices = np.where(label_array == cls)[0][:n_per_class]
        label = class_names.get(cls, str(cls))
        colour = CLASS_COLOURS.get(cls, "black")

        for idx in indices:
            img = face_array[idx]
            if _is_nan_face(img):
                axes[row, 0].axis("off")
                axes[row, 1].axis("off")
                row += 1
                continue

            # Ensure uint8 grayscale
            if img.ndim == 3:
                img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2GRAY)
            else:
                img = img.astype(np.uint8)

            _, hog_img = hog(img, **{**hog_params, "visualize": True})

            axes[row, 0].imshow(img, cmap="gray")
            axes[row, 0].set_title(f"{label} — face", fontsize=8, color=colour)
            axes[row, 0].axis("off")

            axes[row, 1].imshow(hog_img, cmap="magma")
            axes[row, 1].set_title(f"{label} — HOG", fontsize=8, color=colour)
            axes[row, 1].axis("off")

            row += 1

    fig.suptitle("HOG Feature Visualisation", fontsize=13)
    plt.tight_layout()
    plt.show()


def plot_tsne(
    features: np.ndarray,
    label_array: np.ndarray,
    title: str = "t-SNE Feature Space",
    class_names: dict = None,
    perplexity: int = 10,
    random_state: int = 42,
) -> None:
    """
    2-D t-SNE scatter plot of *features* coloured by class.

    A well-separated plot means the feature representation is discriminative
    (different classes cluster apart) and robust (same class forms a tight
    cluster despite image variation).

    Parameters
    ----------
    features     : feature matrix, shape (N, D)
    label_array  : integer labels, shape (N,)
    perplexity   : t-SNE perplexity — with only 80 training samples,
                   keep this low (5–15). Default 10.
    """
    from sklearn.manifold import TSNE

    class_names = _resolve_class_names(class_names)

    print(f"Running t-SNE (perplexity={perplexity}) on {features.shape} ...")
    tsne = TSNE(
        n_components=2, perplexity=perplexity, random_state=random_state, max_iter=1000
    )
    coords = tsne.fit_transform(features)

    fig, ax = plt.subplots(figsize=(8, 6))
    classes = sorted(np.unique(label_array))

    for cls in classes:
        mask = label_array == cls
        colour = CLASS_COLOURS.get(cls, None)
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            c=colour,
            label=class_names.get(cls, str(cls)),
            s=60,
            alpha=0.8,
            edgecolors="white",
            linewidths=0.5,
        )

    ax.legend(fontsize=10, framealpha=0.9)
    ax.set_title(title, fontsize=13)
    ax.set_xlabel("t-SNE dim 1")
    ax.set_ylabel("t-SNE dim 2")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.show()


# =============================================================================
# 4. PCA / eigenface plots
# =============================================================================


def plot_eigenfaces(
    components: np.ndarray,
    face_size: tuple,
    n: int = 16,
    title: str = "Top Eigenfaces",
) -> None:
    """
    Displays the top *n* PCA components reshaped as face images.

    The first few components should resemble blurry face-shaped blobs
    capturing global lighting and pose variation.  Later components capture
    finer structure (eye asymmetry, mouth shape, etc.).

    Parameters
    ----------
    components : PCA components array, shape (n_components, n_pixels)
                 e.g. pca.components_ from sklearn.decomposition.PCA
    face_size  : (height, width) to reshape each component
    n          : how many eigenfaces to show
    """
    n = min(n, len(components))
    ncols = min(8, n)
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(2.2 * ncols, 2.4 * nrows))
    axes = np.array(axes).flatten() if n > 1 else [axes]

    for i, ax in enumerate(axes):
        ax.axis("off")
        if i < n:
            ef = components[i].reshape(face_size)
            # Normalise to [0, 1] for display
            ef = (ef - ef.min()) / (ef.max() - ef.min() + 1e-8)
            ax.imshow(ef, cmap="gray")
            ax.set_title(f"PC {i+1}", fontsize=8)

    fig.suptitle(title, fontsize=13)
    plt.tight_layout()
    plt.show()


def plot_pca_reconstruction(
    original: np.ndarray,
    mean_face: np.ndarray,
    components: np.ndarray,
    face_size: tuple,
    n_components_list: list = None,
) -> None:
    """
    Shows one face reconstructed with an increasing number of PCA components.

    This is the visualisation required by section 2.1.2 of the assignment.
    It demonstrates how reconstruction quality improves with more components
    and helps choose the optimal number *p*.

    Parameters
    ----------
    original         : the flat original image vector, shape (n_pixels,)
    mean_face        : PCA mean vector, shape (n_pixels,)
    components       : PCA components, shape (n_components, n_pixels)
    face_size        : (height, width) to reshape vectors
    n_components_list: list of component counts to try
                       defaults to [1, 5, 10, 20, 40, 60, len(components)]
    """
    if n_components_list is None:
        max_c = len(components)
        n_components_list = [1, 5, 10, 20, 40, 60, max_c]
        n_components_list = [c for c in n_components_list if c <= max_c]

    n = len(n_components_list) + 1  # +1 for original
    fig, axes = plt.subplots(1, n, figsize=(2.5 * n, 3))

    # Original
    axes[0].imshow(original.reshape(face_size), cmap="gray")
    axes[0].set_title("Original", fontsize=9)
    axes[0].axis("off")

    centred = original - mean_face

    for i, k in enumerate(n_components_list, start=1):
        scores = centred @ components[:k].T
        reconstruction = mean_face + scores @ components[:k]
        img = reconstruction.reshape(face_size)
        img = np.clip((img - img.min()) / (img.max() - img.min() + 1e-8), 0, 1)
        axes[i].imshow(img, cmap="gray")
        axes[i].set_title(f"{k} PCs", fontsize=9)
        axes[i].axis("off")

    fig.suptitle("PCA Reconstruction Quality vs Number of Components", fontsize=12)
    plt.tight_layout()
    plt.show()


def plot_pca_scatter(
    scores: np.ndarray,
    label_array: np.ndarray,
    face_crops: np.ndarray = None,
    face_size: tuple = None,
    class_names: dict = None,
    title: str = "Faces projected onto first two Principal Components",
) -> None:
    """
    Scatter plot of all images projected onto PC1 and PC2.

    Optionally overlays thumbnail face crops at each point (like Fig. 2 in
    the assignment brief) when *face_crops* and *face_size* are provided.

    Parameters
    ----------
    scores      : PCA projection matrix, shape (N, n_components)
                  e.g. pca.transform(X)
    label_array : integer labels
    face_crops  : optional grayscale crop array, shape (N, H*W) or (N, H, W)
                  if provided, thumbnails are drawn at each scatter point
    face_size   : (H, W) required when face_crops is provided for reshaping
    """
    class_names = _resolve_class_names(class_names)
    classes = sorted(np.unique(label_array))

    fig, ax = plt.subplots(figsize=(10, 8))

    if face_crops is not None and face_size is not None:
        # Draw thumbnail images at each point (replicates the assignment figure)
        from matplotlib.offsetbox import OffsetImage, AnnotationBbox

        thumb_size = 0.08  # fraction of axis

        for i, (x, y) in enumerate(scores[:, :2]):
            img = face_crops[i]
            if _is_nan_face(img) if img.ndim > 0 else False:
                continue
            img_2d = img.reshape(face_size) if img.ndim == 1 else img
            img_2d = np.clip(
                (img_2d - img_2d.min()) / (img_2d.max() - img_2d.min() + 1e-8), 0, 1
            )
            oi = OffsetImage(img_2d, zoom=thumb_size, cmap="gray")
            ab = AnnotationBbox(oi, (x, y), frameon=False)
            ax.add_artist(ab)
    else:
        # Simple coloured scatter
        for cls in classes:
            mask = label_array == cls
            ax.scatter(
                scores[mask, 0],
                scores[mask, 1],
                c=CLASS_COLOURS.get(cls, None),
                label=class_names.get(cls, str(cls)),
                s=80,
                alpha=0.85,
                edgecolors="white",
                linewidths=0.5,
            )
        ax.legend(fontsize=10)

    # Legend patches (shown even in thumbnail mode)
    patches = [
        mpatches.Patch(
            color=CLASS_COLOURS.get(c, "#888888"), label=class_names.get(c, str(c))
        )
        for c in classes
    ]
    ax.legend(handles=patches, fontsize=10, loc="upper right")

    ax.set_xlabel("Eigenface 1 (PC 1)", fontsize=11)
    ax.set_ylabel("Eigenface 2 (PC 2)", fontsize=11)
    ax.set_title(title, fontsize=13)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.show()


def plot_explained_variance(
    explained_variance_ratio: np.ndarray,
    title: str = "PCA Explained Variance",
) -> None:
    """
    Dual plot: per-component variance (bar) and cumulative variance (line).
    Helps choose the optimal number of components *p*.

    A common choice is the 'elbow' in the curve, or the number of components
    that explain 95 % of total variance.
    """
    cumulative = np.cumsum(explained_variance_ratio)
    n = len(explained_variance_ratio)

    fig, ax1 = plt.subplots(figsize=(10, 4))
    ax2 = ax1.twinx()

    ax1.bar(
        range(1, n + 1),
        explained_variance_ratio * 100,
        color="#4C72B0",
        alpha=0.7,
        label="Per-component variance",
    )
    ax2.plot(
        range(1, n + 1),
        cumulative * 100,
        color="#DD8452",
        linewidth=2,
        label="Cumulative variance",
    )

    # Mark 95 % threshold
    idx_95 = np.searchsorted(cumulative, 0.95)
    ax2.axhline(95, color="red", linestyle="--", linewidth=1, alpha=0.7)
    if idx_95 < n:
        ax2.axvline(idx_95 + 1, color="red", linestyle="--", linewidth=1, alpha=0.7)
        ax2.text(
            idx_95 + 1,
            50,
            f"  {idx_95+1} PCs\n  for 95%",
            color="red",
            fontsize=9,
            va="center",
        )

    ax1.set_xlabel("Principal Component", fontsize=11)
    ax1.set_ylabel("Individual variance (%)", fontsize=11, color="#4C72B0")
    ax2.set_ylabel("Cumulative variance (%)", fontsize=11, color="#DD8452")
    ax1.set_title(title, fontsize=13)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc="center right")

    plt.tight_layout()
    plt.show()
