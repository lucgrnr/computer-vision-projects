# Computer Vision Projects

Two end-to-end computer vision projects, each built around classical and modern
techniques and delivered as a single self-contained, heavily documented notebook.
Every notebook ships with a rendered PDF so the full write-up — code, figures and
results — can be read without running anything.

| Project | What it does |
| --- | --- |
| [**face-detection-recognition**](face-detection-recognition/) | Finds faces in photographs and identifies who they belong to. Compares handcrafted descriptors (HOG), features learned from the data (PCA / eigenfaces) and deep face embeddings, then benchmarks a range of classifiers on top of each representation. |
| [**segmentation-classification-attacks**](segmentation-classification-attacks/) | Understands natural scenes at two levels of granularity — which objects appear in an image, and which object each individual pixel belongs to — then attacks the resulting models with adversarial perturbations to probe how fragile they are. |

Each folder has its own README with the full method breakdown.

## Layout

```
face-detection-recognition/
    face-detection-recognition.ipynb    the project notebook
    face-detection-recognition.pdf      rendered write-up, 76 pages
    datasets/                           image data the notebook reads
segmentation-classification-attacks/
    segmentation-classification-attacks.ipynb
    segmentation-classification-attacks.pdf   rendered write-up, 116 pages
    clip_vitb16_v2_patch.pth            fine-tuned CLIP classifier head
environment.yml                         conda environment for both projects
```

## Getting set up

```bash
conda env create -f environment.yml
conda activate cv
```

`environment.yml` covers both projects. Its header documents an older, stricter
pin set that the second project was originally validated against — worth reading
before you rely on the exact numbers in that notebook.

## A note on large files

`clip_vitb16_v2_patch.pth` (574 MB) is tracked with **Git LFS**, so you need
`git lfs install` before cloning to get the real file rather than a pointer. The
notebook loads it by name to skip retraining; delete it and the training path
runs instead.

Both notebooks resolve their input data through a `DATA_PATH` constant defined
near the top — point it at wherever your copy of the data lives.
