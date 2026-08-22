# Scene Classification, Semantic Segmentation and Adversarial Attacks

Understand natural scenes at two levels of granularity, then deliberately break
the models that do it.

The project works over a subset of PASCAL VOC 2009: everyday scenes annotated
across 20 object categories — people, animals, vehicles, furniture — where an
image usually contains several categories at once, but any single pixel belongs
to exactly one. That split defines the first two parts:

- **Classification** — *which object categories appear anywhere in this image?*
  A multi-label problem: predictions are a set, not a single winner.
- **Segmentation** — *which category does each individual pixel belong to?*
  21 labels, the 20 object categories plus background.

## 1 · Multi-label classification

Three approaches, in increasing order of borrowed knowledge:

- **A small CNN trained from scratch** — the honest baseline, establishing what
  the dataset alone supports without outside data
- **A fine-tuned CLIP ViT-B/16 backbone** — a vision-language model adapted with
  a lightweight multi-label head, trained in two phases: head first with the
  backbone frozen, then partial unfreezing of the top transformer blocks at a
  reduced learning rate
- **DINOv2 features** — self-supervised representations, probed for how much
  spatial structure survives into a globally-pooled embedding

Multi-label brings its own problems, addressed explicitly in the notebook:
severe class imbalance is handled with **asymmetric loss**, which penalises
false negatives and false positives differently; per-class decision thresholds
are tuned on validation rather than left at 0.5; and folds are built with
**multi-label stratification** so rare categories stay represented. Evaluation
uses average precision alongside per-class F1.

## 2 · Semantic segmentation

- **U-Net** — the encoder–decoder baseline, with a Dice-based loss that handles
  the heavy background-versus-object imbalance better than plain cross-entropy
- **Mask2Former** with a Swin transformer backbone — a mask-attention
  architecture that predicts a set of masks and labels rather than classifying
  each pixel independently

Scored with Dice and mean IoU, broken down per class so that failures are
attributable. The write-up examines where predictions break down — thin
structures, object boundaries, and categories that are simply rare in the
training data.

## 3 · Adversarial attacks

The final part attacks the segmentation models under a **white-box** threat
model, where the attacker has full access to weights and gradients. Perturbations
are optimised with projected gradient descent under an ε-bounded constraint, so
the modified images stay close to the originals and remain unremarkable to a
human eye.

Several attack settings are compared:

- **untargeted** — degrade the segmentation output however possible
- **targeted** — force the model to hallucinate a *chosen* category where none
  exists, run against each category in turn to expose which are easiest to forge
- **universal** — a single fixed perturbation trained to transfer across many
  images, rather than one optimised per image

Attacks are run against both Mask2Former and SegFormer, so the results
distinguish weaknesses of a particular network from weaknesses of the general
approach. The discussion covers what the success rates say about deploying such
models, and what defences would cost.

## Running it

Open `segmentation-classification-attacks.ipynb` and set `DATA_PATH` to your copy
of the dataset. Training every model end to end needs a GPU and substantial time;
`clip_vitb16_v2_patch.pth` in this folder is the fine-tuned CLIP checkpoint, and
the notebook loads it by that exact filename to skip retraining. It is stored
with **Git LFS** — run `git lfs install` before cloning, or you'll get a pointer
file instead of weights.

`segmentation-classification-attacks.pdf` is the rendered write-up. Its later
sections show code without outputs: those results were lost before the PDF was
produced, and rerunning the notebook regenerates them.
