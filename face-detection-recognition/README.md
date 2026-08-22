# Face Detection and Recognition

Locate faces in unconstrained photographs, then decide **who** each face belongs
to. The project's real subject is the middle step: how you turn a cropped face
into a vector, and how much that choice matters compared to the classifier you
put on top of it.

## The data

A small subset of the VGG Face dataset, covering four celebrities and
deliberately chosen so that two pairs look alike — which makes the task
considerably harder than the image count suggests:

| Person | Images | Role |
| --- | --- | --- |
| Jesse Eisenberg | 30 | target class |
| Mila Kunis | 30 | target class |
| Michael Cera | 10 | distractor, resembles Eisenberg |
| Sarah Hyland | 10 | distractor, resembles Kunis |

The classifier resolves three labels: *Jesse*, *Mila*, and *neither*. The two
distractors exist purely to punish a model that has learned coarse appearance
cues instead of identity.

## Pipeline

**1 · Detection and extraction.** Two detectors are implemented and compared:

- a **Haar cascade** detector, with a multi-check filter pipeline that validates
  candidate boxes (including an eye-detection pass) to suppress false positives
- a **ResNet-10 SSD** convolutional detector with sharpness-aware scoring

Both cache their model files on first use and download them if missing, so the
notebook runs from a clean checkout. Detected faces are cropped to 100×100 and
carried forward; crops are manually verified and relabelled where the detector
picked the wrong face in a group photo.

**2 · Feature representations.** Three families, evaluated head to head:

- **Handcrafted** — Histogram of Oriented Gradients (HOG), capturing local edge
  orientation structure
- **Learned from the data** — PCA over the face set, i.e. eigenfaces, with the
  number of retained components tuned against reconstruction quality and
  downstream accuracy
- **Learned from other data** — embeddings from pretrained networks, including a
  FaceNet-style Inception-ResNet trained for face recognition, and MobileNetV2

Feature spaces are visualised with **t-SNE** to see whether identities separate
before any classifier is fitted — a fast sanity check on whether the
representation carries the signal at all.

**3 · Classification.** Each representation is benchmarked across SVM (linear and
RBF), logistic regression, k-NN, random forest, MLP, linear discriminant
analysis, Gaussian process and gradient-boosted trees, under cross-validation.

## What the notebook shows

The interesting result is the size of the gap between representations relative to
the gap between classifiers: swapping HOG for a learned face embedding moves
accuracy far more than swapping one classifier for another on a fixed
representation. The closing discussion covers where the pipeline still fails,
most of it traceable to detection errors and to the two lookalike distractors.

## Running it

Open `face-detection-recognition.ipynb`. The `DATA_PATH` constant near the top
selects the data source — set it to `'datasets'` to use the copy committed in
this folder. Detector weights download automatically on first run.

`face-detection-recognition.pdf` is the same notebook rendered with all figures,
if you'd rather read than execute.
