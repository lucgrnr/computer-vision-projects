"""
run_annotator.py
----------------
Standalone face annotation tool — run directly with:

    python run_annotator.py

Controls
--------
    Click + drag  : draw bounding box around the correct face
    ENTER         : confirm and save, move to next image
    R             : redraw — clear current box
    S             : skip this image (keep existing crop)
    Q             : quit and save all annotations

Annotations are saved to manual_annotations.json after every confirmation.
Load them in the notebook with:

    from annotate import FaceAnnotator
    annotator = FaceAnnotator(train, train_X_dnn, train_y_dnn)
    train_X_dnn, train_y_dnn = annotator.apply_corrections(train)
"""

import json
import os

import cv2
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import RectangleSelector

# =============================================================================
# Configuration — edit these
# =============================================================================

# Indices of images to annotate — add/remove as needed
PROBLEM_INDICES = [18, 41, 49, 52, 53, 59, 61]

MULTI_ANNOTATION_INDICES = [18, 32, 34, 77]  # for multi-face annotation

CLASS_NAMES = {0: "Michael_Sarah", 1: "Jesse", 2: "Mila"}

# Where to save annotations
ANNOTATIONS_PATH = "manual_annotations.json"

# Dataset paths — adjust if your structure differs
TRAIN_CSV = "datasets/train_set.csv"
TRAIN_DIR = "datasets/train"

# =============================================================================
# Load data
# =============================================================================


def load_data():
    import pandas as pd

    print("Loading dataset...")
    train = pd.read_csv(TRAIN_CSV, index_col=0)
    train.index = train.index.rename("id")
    train["img"] = [
        cv2.cvtColor(
            np.load(f"{TRAIN_DIR}/train_{i}.npy", allow_pickle=False), cv2.COLOR_BGR2RGB
        )
        for i, _ in train.iterrows()
    ]
    print(f"  Loaded {len(train)} training images.")
    return train


def load_annotations():
    if os.path.exists(ANNOTATIONS_PATH):
        with open(ANNOTATIONS_PATH) as f:
            data = {int(k): v for k, v in json.load(f).items()}
        print(f"  Loaded {len(data)} existing annotations from '{ANNOTATIONS_PATH}'.")
        return data
    return {}


def save_annotations(annotations):
    with open(ANNOTATIONS_PATH, "w") as f:
        json.dump(annotations, f, indent=2)


# =============================================================================
# FaceAnnotator class — used by the notebook to apply saved annotations
# =============================================================================


class FaceAnnotator:
    """
    Lightweight class used in the notebook to apply annotations
    saved by running this script standalone.

    Usage in notebook
    -----------------
        from annotate import FaceAnnotator

        annotator = FaceAnnotator(
            train_df         = train,
            face_array       = train_X_dnn,
            label_array      = train_y_dnn,
            annotations_path = "manual_annotations.json"
        )
        train_X_dnn, train_y_dnn = annotator.apply_corrections(train)
    """

    def __init__(
        self,
        train_df,
        face_array,
        label_array,
        annotations_path="manual_annotations.json",
    ):
        self.train_df = train_df
        self.face_array = face_array
        self.label_array = label_array
        self.annotations_path = annotations_path
        self.annotations = load_annotations()  # reuse the function above

        print("FaceAnnotator ready.")
        print(f"  {len(face_array)} images loaded.")
        print(f"  {len(self.annotations)} annotations found in '{annotations_path}'.")

    def apply_corrections(self, train_df, face_size=(100, 100)):
        """Replaces wrong crops with manually annotated ones."""
        corrected_X = self.face_array.copy()
        n_corrected = 0

        for idx, (x, y, w, h) in self.annotations.items():
            orig_img = train_df.iloc[idx]["img"]
            img_h, img_w = orig_img.shape[:2]

            cx1 = max(0, x)
            cy1 = max(0, y)
            cx2 = min(img_w, x + w)
            cy2 = min(img_h, y + h)

            if cx2 <= cx1 or cy2 <= cy1:
                print(f"  Warning: annotation #{idx} out of bounds — skipped.")
                continue

            crop = orig_img[cy1:cy2, cx1:cx2]
            corrected_X[idx] = cv2.resize(
                crop, face_size, interpolation=cv2.INTER_AREA
            ).astype(np.float32)
            n_corrected += 1

        print(f"apply_corrections: replaced {n_corrected} crops.")
        return corrected_X, self.label_array

    def list_annotations(self):
        """Prints a table of all saved annotations."""
        if not self.annotations:
            print("No annotations found.")
            return
        print(f"{'idx':>5}  {'name':<22}  {'x':>5}  {'y':>5}  {'w':>5}  {'h':>5}")
        print("─" * 56)
        for idx in sorted(self.annotations):
            x, y, w, h = self.annotations[idx]
            name = self.train_df.iloc[idx]["name"]
            print(f"{idx:>5}  {name:<22}  {x:>5}  {y:>5}  {w:>5}  {h:>5}")

    def show_annotations(self, train_df, face_size=(100, 100)):
        """Shows before/after comparison for all annotated images."""
        if not self.annotations:
            print("No annotations found.")
            return

        n = len(self.annotations)
        fig, axes = plt.subplots(n, 3, figsize=(9, 3.2 * n))
        if n == 1:
            axes = axes[np.newaxis, :]

        fig.suptitle(
            "Original + box  /  DNN crop (before)  /  Manual crop (after)", fontsize=11
        )

        for row_i, (idx, (x, y, w, h)) in enumerate(sorted(self.annotations.items())):
            orig_img = train_df.iloc[idx]["img"]
            img_h, img_w = orig_img.shape[:2]

            # Original with box
            axes[row_i, 0].imshow(orig_img)
            axes[row_i, 0].add_patch(
                patches.Rectangle(
                    (x, y), w, h, linewidth=2, edgecolor="lime", facecolor="none"
                )
            )
            axes[row_i, 0].set_title(
                f"#{idx} — {train_df.iloc[idx]['name']}", fontsize=8
            )
            axes[row_i, 0].axis("off")

            # DNN crop before
            existing = self.face_array[idx]
            if np.isnan(existing).any():
                axes[row_i, 1].set_facecolor("#222")
                axes[row_i, 1].text(
                    0.5,
                    0.5,
                    "No face",
                    color="w",
                    ha="center",
                    va="center",
                    transform=axes[row_i, 1].transAxes,
                )
            else:
                axes[row_i, 1].imshow(np.clip(existing / 255.0, 0, 1))
            axes[row_i, 1].set_title("DNN (before)", fontsize=8, color="red")
            axes[row_i, 1].axis("off")

            # Manual crop after
            cx1 = max(0, x)
            cy1 = max(0, y)
            cx2 = min(img_w, x + w)
            cy2 = min(img_h, y + h)
            if cx2 > cx1 and cy2 > cy1:
                crop = cv2.resize(
                    orig_img[cy1:cy2, cx1:cx2], face_size, interpolation=cv2.INTER_AREA
                )
                axes[row_i, 2].imshow(crop)
            axes[row_i, 2].set_title("Manual (after)", fontsize=8, color="green")
            axes[row_i, 2].axis("off")

        plt.tight_layout()
        plt.show()

    def run_multi_annotation(indices):
        """Standalone runner — call from main() or directly."""
        train = load_data()
        annotations = load_multi_annotations()

        print(f"Multi-face annotation for images: {indices}\n")

        for idx in indices:
            row = train.iloc[idx]
            status = "(already annotated)" if idx in annotations else ""
            print(f"Image #{idx} — {row['name']} {status}")

            action, boxes = annotate_multi_face(idx, row, annotations)

            if action == "confirmed":
                annotations[idx] = [list(b["box"]) + [b["label"]] for b in boxes]
                save_multi_annotations(annotations)
                print(f"  Saved {len(boxes)} boxes for image #{idx}")

            elif action == "skipped":
                print("  Skipped.")

            elif action == "quit":
                print("  Session ended.")
                break

        print(f"\nDone. {len(annotations)} images annotated.")

    def expand_dataset(
        self,
        train_df,
        face_size: tuple = (100, 100),
        multi_annotations_path: str = "multi_annotations.json",
    ):
        """
        Expands the face array by extracting additional crops from
        multi-face annotations and appending them with their labels.

        For example: an image of Jesse + Michael annotated with two boxes
        produces two crops — one labelled Jesse (1) and one labelled
        Michael/Sarah (0) — both appended to the training set.

        Parameters
        ----------
        train_df               : DataFrame with 'img' column
        face_size              : resize target for each crop
        multi_annotations_path : JSON file written by run_multi_annotation

        Returns
        -------
        expanded_X : np.ndarray (N + extra, H, W, 3) float32
        expanded_y : np.ndarray (N + extra,) int
        """
        if not os.path.exists(multi_annotations_path):
            print(
                f"No multi-face annotation file found at '{multi_annotations_path}' — nothing to expand."
            )
            return self.face_array, self.label_array

        with open(multi_annotations_path) as f:
            multi_anns = {int(k): v for k, v in json.load(f).items()}

        extra_crops = []
        extra_labels = []

        for idx, entries in multi_anns.items():
            orig_img = train_df.iloc[idx]["img"]
            img_h, img_w = orig_img.shape[:2]

            for entry in entries:
                x, y, w, h, label = entry
                cx1 = max(0, x)
                cy1 = max(0, y)
                cx2 = min(img_w, x + w)
                cy2 = min(img_h, y + h)

                if cx2 <= cx1 or cy2 <= cy1:
                    print(
                        f"  Warning: entry in image #{idx} is out of bounds — skipped."
                    )
                    continue

                crop = cv2.resize(
                    orig_img[cy1:cy2, cx1:cx2], face_size, interpolation=cv2.INTER_AREA
                ).astype(np.float32)

                extra_crops.append(crop)
                extra_labels.append(int(label))
                print(
                    f"  Added crop from image #{idx} — class {label} "
                    f"({CLASS_NAMES.get(int(label), label)})"
                )

        if not extra_crops:
            print("No valid crops found in multi-annotations.")
            return self.face_array, self.label_array

        expanded_X = np.concatenate([self.face_array, np.stack(extra_crops)], axis=0)
        expanded_y = np.concatenate(
            [self.label_array, np.array(extra_labels, dtype=self.label_array.dtype)],
            axis=0,
        )

        print(
            f"\nexpand_dataset: {len(self.face_array)} → {len(expanded_X)} samples "
            f"(+{len(extra_crops)} crops from {len(multi_anns)} images)"
        )

        return expanded_X, expanded_y

    def build_final_dataset(
        self,
        train_df,
        face_size: tuple = (100, 100),
        output_dir: str = "final_train_extracted_faces",
        manual_annotations_path: str = "manual_annotations.json",
        multi_annotations_path: str = "multi_annotations.json",
    ):
        """
        One-call function that:
          1. Applies single-face corrections (manual_annotations.json)
          2. Expands dataset with multi-face crops (multi_annotations.json)
          3. Saves all crops as PNG files to *output_dir* organised by class
          4. Saves train_X.npy / train_y.npy / test_X.npy to *output_dir*
          5. Writes metadata.json recording the source of every crop

        This replaces cells 15-17 in the notebook.

        Parameters
        ----------
        train_df                 : DataFrame with 'img' column
        face_size                : (width, height) for all crops
        output_dir               : root directory to write into
        manual_annotations_path  : single-face correction JSON
        multi_annotations_path   : multi-face expansion JSON

        Returns
        -------
        train_X_final : np.ndarray (N+extra, H, W, 3) float32
        train_y_final : np.ndarray (N+extra,) int
        """
        # ── Step 1: apply single-face corrections ────────────────────
        train_X_final, train_y_final = self.apply_corrections(
            train_df, face_size=face_size
        )
        # Update internal state so expand_dataset builds on corrected array
        self.face_array = train_X_final
        self.label_array = train_y_final

        # ── Step 2: expand with multi-face crops ─────────────────────
        train_X_final, train_y_final = self.expand_dataset(
            train_df,
            face_size=face_size,
            multi_annotations_path=multi_annotations_path,
        )

        # ── Step 3: save PNG crops ───────────────────────────────────
        class_folders = {0: "Michael_Sarah", 1: "Jesse", 2: "Mila"}
        for name in class_folders.values():
            os.makedirs(os.path.join(output_dir, name), exist_ok=True)

        # Load annotation files to tag each crop's source
        manual_anns = {}
        if os.path.exists(manual_annotations_path):
            with open(manual_annotations_path) as f:
                manual_anns = {int(k): v for k, v in json.load(f).items()}

        multi_anns_raw = {}
        if os.path.exists(multi_annotations_path):
            with open(multi_annotations_path) as f:
                multi_anns_raw = {int(k): v for k, v in json.load(f).items()}

        # Build source lookup for extra crops in same order as expand_dataset
        extra_sources = []
        for orig_idx, entries in multi_anns_raw.items():
            img_h, img_w = train_df.iloc[orig_idx]["img"].shape[:2]
            for entry in entries:
                x, y, w, h, label = entry
                if min(img_w, x + w) > max(0, x) and min(img_h, y + h) > max(0, y):
                    extra_sources.append((int(orig_idx), int(label)))

        n_original = len(self.face_array)  # before expansion
        metadata = []

        for idx in range(len(train_X_final)):
            img = train_X_final[idx]
            label = int(train_y_final[idx])
            class_folder = class_folders.get(label, f"class_{label}")

            if idx < n_original:
                orig_idx = idx
                person_name = train_df.iloc[idx]["name"]
                source = "corrected" if idx in manual_anns else "dnn"
            else:
                src_idx = idx - n_original
                orig_idx, _ = extra_sources[src_idx]
                person_name = train_df.iloc[orig_idx]["name"]
                source = "multi_annotation"

            filename = f"{idx}_{person_name}_{source}.png"
            filepath = os.path.join(output_dir, class_folder, filename)

            if not np.isnan(img).any():
                img_bgr = cv2.cvtColor(
                    np.clip(img, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR
                )
                cv2.imwrite(filepath, img_bgr)

            metadata.append(
                {
                    "index": idx,
                    "orig_index": orig_idx,
                    "name": person_name,
                    "class": label,
                    "source": source,
                    "file": os.path.join(class_folder, filename),
                }
            )

        with open(os.path.join(output_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)

        # ── Step 4: save .npy arrays ─────────────────────────────────
        np.save(os.path.join(output_dir, "train_X.npy"), train_X_final)
        np.save(os.path.join(output_dir, "train_y.npy"), train_y_final)

        # ── Summary ──────────────────────────────────────────────────
        sources = {}
        for m in metadata:
            sources[m["source"]] = sources.get(m["source"], 0) + 1

        print(f"\nbuild_final_dataset complete → '{output_dir}/'")
        print(f"  Total crops : {len(train_X_final)}")
        print(f"  By source   : {sources}")
        print(
            f"  By class    : {dict(zip(*np.unique(train_y_final, return_counts=True)))}"
        )
        for folder in class_folders.values():
            n = len(
                [
                    f
                    for f in os.listdir(os.path.join(output_dir, folder))
                    if f.endswith(".png")
                ]
            )
            print(f"    {folder}/  ({n} images)")

        return train_X_final, train_y_final

    def plot_annotation_summary(
        self,
        train_df,
        train_X_dnn_original: np.ndarray,
        train_X_final: np.ndarray,
        train_y_final: np.ndarray,
        manual_annotations_path: str = "manual_annotations.json",
        multi_annotations_path: str = "multi_annotations.json",
        face_size: tuple = (100, 100),
    ) -> None:
        """
        Produces two comparison plots:

        Plot 1 — Single-face corrections
            For each manually corrected image shows:
            original image | DNN crop (before) | manual crop (after)

        Plot 2 — Multi-face expansions
            For each multi-annotated image shows:
            original image with boxes | each extracted crop with its label

        This replaces cells 18-20 in the notebook.

        Parameters
        ----------
        train_df              : DataFrame with 'img' column
        train_X_dnn_original  : unmodified DNN crops — used as the "before" column
        train_X_final         : corrected + expanded array — used as "after"
        train_y_final         : labels aligned with train_X_final
        """
        # Load annotation files
        manual_anns = {}
        if os.path.exists(manual_annotations_path):
            with open(manual_annotations_path) as f:
                manual_anns = {int(k): v for k, v in json.load(f).items()}

        multi_data = {}
        if os.path.exists(multi_annotations_path):
            with open(multi_annotations_path) as f:
                multi_data = {int(k): v for k, v in json.load(f).items()}

        changed_indices = sorted(manual_anns.keys())
        multi_orig_indices = sorted(multi_data.keys())

        print(f"Corrected images : {changed_indices}")
        print(f"Multi-face images: {multi_orig_indices}")

        # ── Plot 1: single-face corrections ──────────────────────────
        n = len(changed_indices)
        if n > 0:
            fig, axes = plt.subplots(n, 3, figsize=(10, 3.5 * n))
            if n == 1:
                axes = axes[np.newaxis, :]

            fig.suptitle(
                "Single-face corrections\n"
                "Original image  |  DNN crop (before)  |  Manual crop (after)",
                fontsize=12,
            )

            for row_i, idx in enumerate(changed_indices):
                orig_img = train_df.iloc[idx]["img"]

                # Original image
                axes[row_i, 0].imshow(orig_img)
                axes[row_i, 0].set_title(
                    f"#{idx} — {train_df.iloc[idx]['name']}", fontsize=8
                )
                axes[row_i, 0].axis("off")

                # DNN crop before correction — use the preserved original
                dnn_crop = train_X_dnn_original[idx]
                if np.isnan(dnn_crop).any():
                    axes[row_i, 1].set_facecolor("#222")
                    axes[row_i, 1].text(
                        0.5,
                        0.5,
                        "No face",
                        color="w",
                        ha="center",
                        va="center",
                        transform=axes[row_i, 1].transAxes,
                    )
                else:
                    axes[row_i, 1].imshow(np.clip(dnn_crop / 255.0, 0, 1))
                axes[row_i, 1].set_title("DNN (before)", fontsize=8, color="red")
                axes[row_i, 1].axis("off")

                # Final crop after correction
                final_crop = train_X_final[idx]
                axes[row_i, 2].imshow(np.clip(final_crop / 255.0, 0, 1))
                axes[row_i, 2].set_title("Manual (after)", fontsize=8, color="green")
                axes[row_i, 2].axis("off")

            plt.tight_layout()
            plt.show()

        # ── Plot 2: multi-face expansions ─────────────────────────────
        label_colours = {
            0: "#DD8452",
            1: "#55A868",
            2: "#4C72B0",
            3: "#E15759",
            4: "#76B7B2",
        }
        label_names = {
            0: "Jesse",
            1: "Mila",
            2: "Michael",
            3: "Sarah",
            4: "Other",
        }

        for orig_idx, entries in multi_data.items():
            orig_img = train_df.iloc[orig_idx]["img"]
            n_entries = len(entries)
            img_h, img_w = orig_img.shape[:2]

            fig, axes = plt.subplots(1, 1 + n_entries, figsize=(4 * (1 + n_entries), 4))
            # Ensure axes is always a list
            if 1 + n_entries == 2:
                axes = list(axes)

            fig.suptitle(
                f"Multi-face expansion — Image #{orig_idx} "
                f"({train_df.iloc[orig_idx]['name']})\n"
                f"Original  |  {n_entries} face(s) added to dataset",
                fontsize=11,
            )

            # Original image with all boxes
            axes[0].imshow(orig_img)
            for entry in entries:
                x, y, w, h, label = entry
                colour = label_colours.get(label, "white")
                name = label_names.get(label, str(label))
                axes[0].add_patch(
                    patches.Rectangle(
                        (x, y), w, h, linewidth=2, edgecolor=colour, facecolor="none"
                    )
                )
                axes[0].text(
                    x, y - 6, name, color=colour, fontsize=8, fontweight="bold"
                )
            axes[0].set_title("Original + annotation boxes", fontsize=9)
            axes[0].axis("off")

            # Each extracted crop
            for col, entry in enumerate(entries, start=1):
                x, y, w, h, label = entry
                cx1 = max(0, x)
                cy1 = max(0, y)
                cx2 = min(img_w, x + w)
                cy2 = min(img_h, y + h)
                colour = label_colours.get(label, "white")
                name = label_names.get(label, str(label))

                if cx2 > cx1 and cy2 > cy1:
                    axes[col].imshow(orig_img[cy1:cy2, cx1:cx2])
                axes[col].set_title(
                    f"Added as class {label}\n({name})", fontsize=9, color=colour
                )
                axes[col].axis("off")

            plt.tight_layout()
            plt.show()


# =============================================================================
# Annotation UI
# =============================================================================


def annotate_image(idx, row, annotations):
    """
    Opens a matplotlib window for image *idx*.
    Returns 'confirmed', 'skipped', or 'quit'.
    """
    orig_img = row["img"]
    img_h, img_w = orig_img.shape[:2]

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    fig.patch.set_facecolor("#1a1a1a")
    plt.subplots_adjust(bottom=0.12)

    fig.suptitle(
        f"Image #{idx}  |  {row['name']}  |  {img_w}x{img_h}px\n"
        f"ENTER = confirm    R = redraw    S = skip    Q = quit",
        color="white",
        fontsize=11,
    )

    # Left: original image with coordinate grid
    ax_img = axes[0]
    ax_img.imshow(orig_img)
    ax_img.set_facecolor("#111")

    # Grid every 50px so coordinates are easy to estimate visually
    for x in range(0, img_w, 50):
        ax_img.axvline(x, color="white", alpha=0.2, linewidth=0.5)
        ax_img.text(x + 2, 10, str(x), color="white", fontsize=6, alpha=0.7)
    for y in range(0, img_h, 50):
        ax_img.axhline(y, color="white", alpha=0.2, linewidth=0.5)
        ax_img.text(5, y + 12, str(y), color="white", fontsize=6, alpha=0.7)

    # Show previous annotation in yellow if it exists
    if idx in annotations:
        px, py, pw, ph = annotations[idx]
        prev = patches.Rectangle(
            (px, py),
            pw,
            ph,
            linewidth=2,
            edgecolor="yellow",
            facecolor="none",
            linestyle="--",
            label="Previous annotation",
        )
        ax_img.add_patch(prev)
        ax_img.legend(
            fontsize=8, facecolor="#333", labelcolor="white", loc="upper right"
        )

    ax_img.set_title(
        "Original image — draw box around the correct face", color="white", fontsize=9
    )
    ax_img.axis("off")

    # Right: face crop preview (updated as user draws)
    ax_crop = axes[1]
    ax_crop.set_facecolor("#222")
    ax_crop.text(
        0.5,
        0.5,
        "Draw a box\nto preview crop",
        color="#888",
        ha="center",
        va="center",
        transform=ax_crop.transAxes,
        fontsize=12,
    )
    ax_crop.set_title("Crop preview", color="white", fontsize=9)
    ax_crop.axis("off")

    # Shared state
    state = {"box": None, "action": None}

    def on_select(eclick, erelease):
        x1 = min(eclick.xdata, erelease.xdata)
        y1 = min(eclick.ydata, erelease.ydata)
        x2 = max(eclick.xdata, erelease.xdata)
        y2 = max(eclick.ydata, erelease.ydata)

        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        w, h = x2 - x1, y2 - y1

        if w < 5 or h < 5:
            return

        state["box"] = (x1, y1, w, h)

        # Remove old green box
        for p in [
            p
            for p in ax_img.patches
            if p.get_edgecolor()[1] > 0.8 and p.get_edgecolor()[0] < 0.5
        ]:
            p.remove()

        # Draw new green box
        rect = patches.Rectangle(
            (x1, y1), w, h, linewidth=2, edgecolor="lime", facecolor="none"
        )
        ax_img.add_patch(rect)

        # Update crop preview on the right
        cx1 = max(0, x1)
        cy1 = max(0, y1)
        cx2 = min(img_w, x2)
        cy2 = min(img_h, y2)
        if cx2 > cx1 and cy2 > cy1:
            crop = orig_img[cy1:cy2, cx1:cx2]
            ax_crop.clear()
            ax_crop.imshow(crop)
            ax_crop.set_title(
                f"Crop preview\nx={x1} y={y1} w={w} h={h}", color="lime", fontsize=9
            )
            ax_crop.axis("off")

        fig.canvas.draw_idle()

    def on_key(event):
        key = event.key

        if key == "enter":
            if state["box"] is None:
                print("  Draw a box first, then press ENTER.")
                return
            state["action"] = "confirmed"
            plt.close(fig)

        elif key == "r":
            state["box"] = None
            # Remove all green boxes
            to_remove = [
                p
                for p in ax_img.patches
                if list(p.get_edgecolor()[:3]) != [1.0, 1.0, 0.0]
            ]
            for p in to_remove:
                p.remove()
            ax_crop.clear()
            ax_crop.set_facecolor("#222")
            ax_crop.text(
                0.5,
                0.5,
                "Draw a box\nto preview crop",
                color="#888",
                ha="center",
                va="center",
                transform=ax_crop.transAxes,
                fontsize=12,
            )
            ax_crop.set_title("Crop preview", color="white", fontsize=9)
            ax_crop.axis("off")
            fig.canvas.draw_idle()
            print("  Box cleared — draw again.")

        elif key == "s":
            state["action"] = "skipped"
            plt.close(fig)

        elif key == "q":
            state["action"] = "quit"
            plt.close(fig)

    selector = RectangleSelector(
        ax_img,
        on_select,
        useblit=True,
        button=[1],
        minspanx=5,
        minspany=5,
        spancoords="pixels",
        interactive=True,
    )

    fig.canvas.mpl_connect("key_press_event", on_key)
    plt.show()  # BLOCKING — waits here until window is closed

    return state["action"], state["box"]


# =============================================================================
# Multi-face annotation — add to annotate.py
# =============================================================================

MULTI_ANNOTATIONS_PATH = "multi_annotations.json"


def load_multi_annotations():
    if os.path.exists(MULTI_ANNOTATIONS_PATH):
        with open(MULTI_ANNOTATIONS_PATH) as f:
            return {int(k): v for k, v in json.load(f).items()}
    return {}


def save_multi_annotations(annotations):
    with open(MULTI_ANNOTATIONS_PATH, "w") as f:
        json.dump(annotations, f, indent=2)


EXTRACTED_LABELS_PATH = "extracted_labels.json"


def load_extracted_labels(path=EXTRACTED_LABELS_PATH):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def save_extracted_labels(labels, path=EXTRACTED_LABELS_PATH):
    with open(path, "w") as f:
        json.dump(labels, f, indent=2)


def label_extracted_face(image_path, existing_label=None):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Unable to load image '{image_path}'")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    LABEL_NAMES = {
        0: "Jesse",
        1: "Mila",
        2: "Michael",
        3: "Sarah",
        4: "Other",
    }

    fig, ax = plt.subplots(figsize=(6, 6))
    fig.patch.set_facecolor("#1a1a1a")
    ax.imshow(img)
    ax.axis("off")

    title = os.path.basename(image_path)
    if existing_label is not None:
        title += f"  | current: {LABEL_NAMES.get(existing_label, existing_label)}"
    title += "\nPress 0=Jesse 1=Mila 2=Michael 3=Sarah 4=Other  " "S=skip  Q=quit"
    fig.suptitle(title, color="white", fontsize=10)

    state = {"action": None, "label": existing_label}

    def on_key(event):
        key = event.key
        if key in ("0", "1", "2", "3", "4"):
            state["label"] = int(key)
            state["action"] = "confirmed"
            plt.close(fig)
        elif key == "s":
            state["action"] = "skipped"
            plt.close(fig)
        elif key == "q":
            state["action"] = "quit"
            plt.close(fig)

    fig.canvas.mpl_connect("key_press_event", on_key)
    plt.show()

    return state["action"], state["label"]


def run_extracted_labeling(folder, labels_path=EXTRACTED_LABELS_PATH):
    labels = load_extracted_labels(labels_path)
    image_paths = []

    for root, _, files in os.walk(folder):
        for filename in sorted(files):
            if filename.lower().endswith(
                (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
            ):
                image_paths.append(os.path.join(root, filename))

    if not image_paths:
        print(f"No images found in '{folder}'")
        return

    print(f"Labeling {len(image_paths)} images from '{folder}'")
    for image_path in image_paths:
        key = os.path.relpath(image_path, folder)
        existing = labels.get(key)
        status = f"(current={existing})" if existing is not None else ""
        print(f"Image: {key} {status}")

        action, label = label_extracted_face(image_path, existing)
        if action == "confirmed":
            labels[key] = label
            save_extracted_labels(labels, labels_path)
            print(f"  Saved label {label} for {key}")
        elif action == "skipped":
            print("  Skipped.")
        elif action == "quit":
            print("  Session ended.")
            break

    print(f"Done. {len(labels)} images labeled in '{labels_path}'")


def save_labeled_extracted_faces(
    source_folder,
    labels_path=EXTRACTED_LABELS_PATH,
    output_dir="labeled_extracted_faces",
):
    labels = load_extracted_labels(labels_path)
    if not labels:
        print(f"No labels found in '{labels_path}'")
        return

    LABEL_NAMES = {
        0: "Jesse",
        1: "Mila",
        2: "Michael",
        3: "Sarah",
        4: "Other",
    }

    for name in LABEL_NAMES.values():
        os.makedirs(os.path.join(output_dir, name), exist_ok=True)

    saved = 0
    missing = 0
    metadata = []

    for relpath, label in labels.items():
        src_path = os.path.join(source_folder, relpath)
        if not os.path.exists(src_path):
            missing += 1
            print(f"Missing image: {src_path}")
            continue

        class_folder = LABEL_NAMES.get(label, f"class_{label}")
        dst_dir = os.path.join(output_dir, class_folder)
        os.makedirs(dst_dir, exist_ok=True)

        dst_path = os.path.join(dst_dir, os.path.basename(relpath))
        img = cv2.imread(src_path)
        if img is None:
            missing += 1
            print(f"Unable to read image: {src_path}")
            continue

        cv2.imwrite(dst_path, img)
        saved += 1
        metadata.append(
            {
                "source": src_path,
                "file": os.path.join(class_folder, os.path.basename(relpath)),
                "label": label,
                "class_name": LABEL_NAMES.get(label, str(label)),
            }
        )

    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(
        f"Saved {saved} labeled images to '{output_dir}'"
        + (f" ({missing} missing)" if missing else "")
    )


def annotate_multi_face(idx, row, existing_annotations):
    """
    Opens a matplotlib window for image *idx* and lets you draw
    multiple bounding boxes, each with a manually assigned label.

    Controls
    --------
    Click + drag    : draw a bounding box
    0 / 1 / 2 / 3 / 4 : assign label to the LAST drawn box
                        0 = Jesse, 1 = Mila, 2 = Michael, 3 = Sarah, 4 = Other
    ENTER           : save all boxes and move to next image
    R               : remove the last drawn box
    C               : clear all boxes
    Q               : quit
    """
    orig_img = row["img"]
    img_h, img_w = orig_img.shape[:2]

    fig, ax = plt.subplots(figsize=(12, 8))
    fig.patch.set_facecolor("#1a1a1a")

    fig.suptitle(
        f"Image #{idx}  |  {row['name']}  |  {img_w}x{img_h}px\n"
        f"Draw boxes, then press 0/1/2/3/4 to label last box  "
        f"|  ENTER=save  R=remove last  C=clear all  Q=quit",
        color="white",
        fontsize=10,
    )

    ax.imshow(orig_img)
    ax.set_facecolor("#111")

    # Grid every 50px
    for x in range(0, img_w, 50):
        ax.axvline(x, color="white", alpha=0.2, linewidth=0.5)
        ax.text(x + 2, 10, str(x), color="white", fontsize=6, alpha=0.7)
    for y in range(0, img_h, 50):
        ax.axhline(y, color="white", alpha=0.2, linewidth=0.5)
        ax.text(5, y + 12, str(y), color="white", fontsize=6, alpha=0.7)

    ax.set_title("Draw a box around each face", color="white", fontsize=9)
    ax.axis("off")

    # Label colours and names
    LABEL_COLOURS = {
        0: "#DD8452",
        1: "#55A868",
        2: "#4C72B0",
        3: "#E15759",
        4: "#76B7B2",
    }
    LABEL_NAMES = {
        0: "Jesse",
        1: "Mila",
        2: "Michael",
        3: "Sarah",
        4: "Other",
    }

    # Show existing annotations for this image
    if idx in existing_annotations:
        for entry in existing_annotations[idx]:
            x, y, w, h, label = entry
            colour = LABEL_COLOURS.get(label, "white")
            name = LABEL_NAMES.get(label, str(label))
            ax.add_patch(
                patches.Rectangle(
                    (x, y),
                    w,
                    h,
                    linewidth=2,
                    edgecolor=colour,
                    facecolor="none",
                    linestyle="--",
                )
            )
            ax.text(
                x,
                y - 5,
                f"{name} ({label})",
                color=colour,
                fontsize=8,
                fontweight="bold",
            )

    # State: list of dicts {box, label, patch, text}
    state = {"boxes": [], "action": None}

    # Legend
    legend_text = ax.text(
        0.01,
        0.01,
        "No boxes yet",
        transform=ax.transAxes,
        color="white",
        fontsize=8,
        verticalalignment="bottom",
        bbox=dict(facecolor="#333", alpha=0.8, pad=4),
    )

    def update_legend():
        if not state["boxes"]:
            legend_text.set_text("No boxes yet")
            return
        lines = []
        for i, b in enumerate(state["boxes"]):
            lbl = b["label"]
            name = (
                LABEL_NAMES.get(lbl, "unlabelled") if lbl is not None else "unlabelled"
            )
            colour = LABEL_COLOURS.get(lbl, "white")
            lines.append(f"Box {i+1}: {name}")
        legend_text.set_text("\n".join(lines))
        fig.canvas.draw_idle()

    def on_select(eclick, erelease):
        x1 = int(min(eclick.xdata, erelease.xdata))
        y1 = int(min(eclick.ydata, erelease.ydata))
        x2 = int(max(eclick.xdata, erelease.xdata))
        y2 = int(max(eclick.ydata, erelease.ydata))
        w, h = x2 - x1, y2 - y1

        if w < 5 or h < 5:
            return

        # Draw box in white (unlabelled) until user presses 0/1/2/3/4
        patch = patches.Rectangle(
            (x1, y1), w, h, linewidth=2, edgecolor="white", facecolor="none"
        )
        ax.add_patch(patch)

        # Add label text placeholder
        txt = ax.text(
            x1, y1 - 5, "unlabelled", color="white", fontsize=8, fontweight="bold"
        )

        state["boxes"].append(
            {
                "box": (x1, y1, w, h),
                "label": None,
                "patch": patch,
                "text": txt,
            }
        )
        update_legend()
        print(
            f"  Box {len(state['boxes'])} drawn: x={x1} y={y1} w={w} h={h}"
            f" — press 0, 1, 2, 3, or 4 to label it"
        )

    def on_key(event):
        key = event.key

        # Label the last drawn box
        if key in ("0", "1", "2", "3", "4"):
            if not state["boxes"]:
                print("  Draw a box first.")
                return
            label = int(key)
            colour = LABEL_COLOURS[label]
            name = LABEL_NAMES[label]

            last = state["boxes"][-1]
            last["label"] = label
            last["patch"].set_edgecolor(colour)
            last["text"].set_text(name)
            last["text"].set_color(colour)
            update_legend()
            fig.canvas.draw_idle()
            print(f"  Box {len(state['boxes'])} labelled as {name}")

        elif key == "r":
            # Remove last box
            if not state["boxes"]:
                return
            last = state["boxes"].pop()
            last["patch"].remove()
            last["text"].remove()
            update_legend()
            fig.canvas.draw_idle()
            print("  Last box removed.")

        elif key == "c":
            # Clear all boxes
            for b in state["boxes"]:
                b["patch"].remove()
                b["text"].remove()
            state["boxes"].clear()
            update_legend()
            fig.canvas.draw_idle()
            print("  All boxes cleared.")

        elif key == "enter":
            unlabelled = [b for b in state["boxes"] if b["label"] is None]
            if unlabelled:
                print(
                    f"  {len(unlabelled)} box(es) have no label — "
                    f"press 0/1/2/3/4 to label them before confirming."
                )
                return
            if not state["boxes"]:
                print("  No boxes drawn — press S to skip or draw boxes first.")
                return
            state["action"] = "confirmed"
            plt.close(fig)

        elif key == "s":
            state["action"] = "skipped"
            plt.close(fig)

        elif key == "q":
            state["action"] = "quit"
            plt.close(fig)

    selector = RectangleSelector(
        ax,
        on_select,
        useblit=True,
        button=[1],
        minspanx=5,
        minspany=5,
        spancoords="pixels",
        interactive=True,
    )

    fig.canvas.mpl_connect("key_press_event", on_key)
    plt.show()  # blocking

    # Extract results
    result = [
        {"box": b["box"], "label": b["label"]}
        for b in state["boxes"]
        if b["label"] is not None
    ]
    return state["action"], result


def run_multi_annotation(indices):
    """Standalone runner — call from main() or directly."""
    train = load_data()
    annotations = load_multi_annotations()

    print(f"Multi-face annotation for images: {indices}\n")

    for idx in indices:
        row = train.iloc[idx]
        status = "(already annotated)" if idx in annotations else ""
        print(f"Image #{idx} — {row['name']} {status}")

        action, boxes = annotate_multi_face(idx, row, annotations)

        if action == "confirmed":
            annotations[idx] = [list(b["box"]) + [b["label"]] for b in boxes]
            save_multi_annotations(annotations)
            print(f"  Saved {len(boxes)} boxes for image #{idx}")

        elif action == "skipped":
            print("  Skipped.")

        elif action == "quit":
            print("  Session ended.")
            break

    print(f"\nDone. {len(annotations)} images annotated.")


# =============================================================================
# Main loop
# =============================================================================


def main():
    train = load_data()
    annotations = load_annotations()
    multi_annotations = load_multi_annotations()

    # ── Single-face corrections ──────────────────────────────────────
    print(f"\nSingle-face annotation for images: {PROBLEM_INDICES}")
    print("Controls: ENTER=confirm  R=redraw  S=skip  Q=quit\n")

    for i, idx in enumerate(PROBLEM_INDICES):
        row = train.iloc[idx]
        status = "(already annotated)" if idx in annotations else ""
        print(f"[{i+1}/{len(PROBLEM_INDICES)}] Image #{idx} — {row['name']} {status}")

        action, box = annotate_image(idx, row, annotations)

        if action == "confirmed":
            annotations[idx] = box
            save_annotations(annotations)
            print(f"  Saved: x={box[0]} y={box[1]} w={box[2]} h={box[3]}")
        elif action == "skipped":
            print("  Skipped.")
        elif action == "quit":
            print("  Session ended early.")
            break

    # ── Multi-face expansion ─────────────────────────────────────────
    print(f"\nMulti-face annotation for images: {MULTI_ANNOTATION_INDICES}")
    print(
        "Controls: draw box → 0/1/2/3/4 to label → ENTER=save  R=remove last  C=clear  Q=quit\n"
    )

    for i, idx in enumerate(MULTI_ANNOTATION_INDICES):
        row = train.iloc[idx]
        status = "(already annotated)" if idx in multi_annotations else ""
        print(
            f"[{i+1}/{len(MULTI_ANNOTATION_INDICES)}] Image #{idx} — {row['name']} {status}"
        )

        action, boxes = annotate_multi_face(idx, row, multi_annotations)

        if action == "confirmed":
            multi_annotations[idx] = [list(b["box"]) + [b["label"]] for b in boxes]
            save_multi_annotations(multi_annotations)
            print(f"  Saved {len(boxes)} boxes for image #{idx}")
        elif action == "skipped":
            print("  Skipped.")
        elif action == "quit":
            print("  Session ended early.")
            break

    print("\nDone.")
    print(f"  {len(annotations)} single-face corrections in '{ANNOTATIONS_PATH}'")
    print(
        f"  {len(multi_annotations)} multi-face annotations in '{MULTI_ANNOTATIONS_PATH}'"
    )


if __name__ == "__main__":
    main()
