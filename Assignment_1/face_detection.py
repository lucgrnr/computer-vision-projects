"""
face_detection.py
-----------------
Face detection module for KUL H02A5a Computer Vision Group Assignment 1.

Provides two detectors:
    - HAARPreprocessor  : classical Haar cascade with 7-check filter pipeline
    - DNNFaceDetector   : ResNet SSD detector with sharpness-aware scoring

Typical usage
-------------
    from face_detection import DNNFaceDetector, HAARPreprocessor

    preprocessor = DNNFaceDetector(path='tmp', face_size=(100, 100))
    train_X = preprocessor(train)   # train is a DataFrame with an 'img' column
    test_X  = preprocessor(test)
"""

import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib import request

import cv2
import numpy as np


# =============================================================================
# Shared utilities
# =============================================================================


def _download_file(url: str, dest_path: str) -> None:
    """Downloads a file from *url* and writes it to *dest_path*."""
    print(f"  Downloading {os.path.basename(dest_path)} ...")
    with request.urlopen(url) as r, open(dest_path, "wb") as f:
        f.write(r.read())
    print(f"  Saved to {dest_path}")


def _nan_face(face_size: tuple) -> np.ndarray:
    """Returns an (H, W, 3) float32 array filled with NaN — used as a sentinel
    when no face is detected so downstream code can detect and skip it."""
    img = np.empty(face_size + (3,), dtype=np.float32)
    img[:] = np.nan
    return img


def _center_crop_fallback(img: np.ndarray, face_size: tuple) -> np.ndarray:
    """
    Last-resort fallback when detection finds nothing.

    Crops the largest centered square from *img* and resizes to *face_size*.
    Portrait photographers almost always center their subject, so a center
    crop is a reasonable guess and far better than returning NaN which gives
    the classifier nothing to work with.
    """
    h, w = img.shape[:2]
    size = min(h, w)
    y0 = (h - size) // 2
    x0 = (w - size) // 2
    crop = img[y0 : y0 + size, x0 : x0 + size]
    return cv2.resize(crop, face_size, interpolation=cv2.INTER_AREA).astype(np.float32)


# =============================================================================
# HAAR Preprocessor
# =============================================================================


class HAARPreprocessor:
    """
    Classical Haar cascade face detector with an extended false-positive
    filter pipeline and sharpness-aware face selection.

    Detection pipeline
    ------------------
    1. Haar cascade (strict params) → filter → keep survivors
    2. If nothing survives, relax params (scaleFactor=1.05, minNeighbors=3)
       and filter again.
    3. If still nothing, return center-crop fallback.

    Filter checks (ordered cheapest → most expensive)
    --------------------------------------------------
    1. Aspect ratio          0.7 – 1.4   (faces are roughly square)
    2. Minimum relative size ≥ 10 % of image width  (kills tiny BG faces)
    3. Vertical position     center in top 85 % of image
    4. Laplacian variance    ≥ 100       (rejects flat/uniform patches)
    5. Raw brightness        mean ≥ 50   (rejects genuinely dark regions)
    6. Skin tone             ≥ 15 % skin pixels in HSV space
    7. Colour variance       HSV-saturation std ≥ 20  (rejects plain walls/shirts)

    Face selection (when multiple candidates pass)
    -----------------------------------------------
    score = area × center_weight × eye_bonus × sharpness²

    Parameters
    ----------
    path         : directory to cache cascade XML files
    face_size    : output crop size as (width, height) tuple
    scale_factor : Haar scaleFactor (default 1.1)
    min_neighbors: Haar minNeighbors (default 5)
    n_jobs       : parallel workers (default = CPU count, max 8)
    """

    # Haar cascade URLs (OpenCV GitHub)
    _FACE_URL = (
        "https://raw.githubusercontent.com/opencv/opencv/master/data/"
        "haarcascades/haarcascade_frontalface_default.xml"
    )
    _EYE_URL = (
        "https://raw.githubusercontent.com/opencv/opencv/master/data/"
        "haarcascades/haarcascade_eye.xml"
    )

    def __init__(
        self,
        path: str,
        face_size: tuple,
        scale_factor: float = 1.1,
        min_neighbors: int = 5,
        n_jobs: int | None = None,
    ) -> None:
        self.face_size = face_size
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors
        self.n_jobs = n_jobs or min(os.cpu_count() or 4, 8)

        os.makedirs(path, exist_ok=True)

        self._face_xml = os.path.join(path, "haarcascade_frontalface_default.xml")
        self._eye_xml = os.path.join(path, "haarcascade_eye.xml")

        if not os.path.exists(self._face_xml):
            _download_file(self._FACE_URL, self._face_xml)
        if not os.path.exists(self._eye_xml):
            _download_file(self._EYE_URL, self._eye_xml)

        # Thread-local storage for classifier instances.
        # CascadeClassifier is NOT thread-safe when shared — each thread
        # gets its own instance created on first use and reused thereafter.
        self._tls = threading.local()

    # ------------------------------------------------------------------
    # Thread-local classifier access
    # ------------------------------------------------------------------

    def _get_classifiers(self):
        if not hasattr(self._tls, "face_clf"):
            self._tls.face_clf = cv2.CascadeClassifier(self._face_xml)
            self._tls.eye_clf = cv2.CascadeClassifier(self._eye_xml)
        return self._tls.face_clf, self._tls.eye_clf

    # ------------------------------------------------------------------
    # Filter helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _has_skin_tone(patch_rgb: np.ndarray) -> bool:
        """
        Returns True when ≥ 15 % of the patch pixels fall in the HSV
        skin-tone range.  Rejects cartoons, printed posters, and dark
        fabric that passed the brightness check due to fold edges.
        """
        hsv = cv2.cvtColor(patch_rgb, cv2.COLOR_RGB2HSV)
        lower = np.array([0, 30, 50], dtype=np.uint8)
        upper = np.array([25, 170, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower, upper)
        return mask.sum() / 255.0 / mask.size >= 0.15

    @staticmethod
    def _has_colour_variance(patch_rgb: np.ndarray) -> bool:
        """
        Returns True when the HSV saturation channel std ≥ 20.

        Real faces contain colour variation (lips, eyes, highlights,
        shadows).  Uniform surfaces — beige walls, plain shirts — pass
        the skin-tone hue check but have very low saturation variance.
        Threshold of 20 was chosen from empirical measurements:
        walls/shirts cluster below 15, real faces above 25.
        """
        hsv = cv2.cvtColor(patch_rgb, cv2.COLOR_RGB2HSV)
        return float(hsv[:, :, 1].std()) >= 20.0

    def _count_valid_eyes(self, crop_gray: np.ndarray, eye_clf) -> int:
        """
        Counts eyes in the TOP HALF of *crop_gray* and validates their
        vertical position.

        Searching only the top half prevents the detector locking on
        nostrils or mouth as false eyes.

        Brow ridges — the main false-positive for forehead crops — appear
        at < 15 % of crop height.  Real eyes sit between 20 % and 60 %.

        Returns 0, 1, or 2.
        """
        h = crop_gray.shape[0]
        top_half = crop_gray[: h // 2, :]
        eyes = eye_clf.detectMultiScale(
            top_half, scaleFactor=1.1, minNeighbors=4, minSize=(15, 15)
        )
        if len(eyes) == 0:
            return 0
        valid = [e for e in eyes if 0.20 <= (e[1] + e[3] / 2.0) / h <= 0.60]
        return len(valid)

    # ------------------------------------------------------------------
    # Main filter
    # ------------------------------------------------------------------

    def _filter(self, raw_faces, img, img_gray, img_h, img_w, eye_clf):
        """
        Applies all 7 checks to *raw_faces* and returns a list of
        (x, y, w, h, n_eyes, lap_var) tuples for surviving candidates.
        lap_var is stored here so _best_face can reuse it without
        recomputing the Laplacian.
        """
        survivors = []
        for x, y, w, h in raw_faces:
            # 1. Aspect ratio
            if not (0.7 <= w / float(h) <= 1.4):
                continue
            # 2. Minimum relative size
            if w / float(img_w) < 0.10:
                continue
            # 3. Vertical position of face centre
            if (y + h / 2) / float(img_h) > 0.85:
                continue

            patch_gray = img_gray[y : y + h, x : x + w]
            patch_raw = img[y : y + h, x : x + w]

            # 4. Texture variance (on equalized gray — good edge sensitivity)
            lap_var = cv2.Laplacian(patch_gray, cv2.CV_64F).var()
            if lap_var < 100:
                continue
            # 5. Brightness on RAW patch (equalized gray masks dark crops)
            if patch_raw.mean() < 50:
                continue
            # 6. Skin tone
            if not self._has_skin_tone(patch_raw):
                continue
            # 7. Colour variance
            if not self._has_colour_variance(patch_raw):
                continue

            n_eyes = self._count_valid_eyes(patch_gray, eye_clf)
            survivors.append((x, y, w, h, n_eyes, lap_var))

        return survivors

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------

    def detect_faces(self, img: np.ndarray):
        """
        Two-stage detection with automatic fallback.

        Returns a list of (x, y, w, h, n_eyes, lap_var) tuples that
        passed all filter checks.  Empty list when nothing survives.
        """
        face_clf, eye_clf = self._get_classifiers()
        img_h, img_w = img.shape[:2]
        img_gray = cv2.equalizeHist(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY))

        # Stage 1 — strict
        raw = face_clf.detectMultiScale(
            img_gray,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE,
        )
        faces = self._filter(
            raw if len(raw) > 0 else [], img, img_gray, img_h, img_w, eye_clf
        )

        # Stage 2 — relaxed fallback
        if not faces:
            raw = face_clf.detectMultiScale(
                img_gray,
                scaleFactor=1.05,
                minNeighbors=3,
                minSize=(30, 30),
                flags=cv2.CASCADE_SCALE_IMAGE,
            )
            faces = self._filter(
                raw if len(raw) > 0 else [], img, img_gray, img_h, img_w, eye_clf
            )

        return faces

    # ------------------------------------------------------------------
    # Face selection
    # ------------------------------------------------------------------

    def _best_face(self, faces, img_h: int, img_w: int):
        """
        Returns the (x, y, w, h) of the highest-scoring candidate.

        Score = area × center_weight × eye_bonus × sharpness²

        area          : fraction of image covered → favours prominent faces
        center_weight : Gaussian decay from rule-of-thirds target (35 % from
                        top, horizontally centred) → portrait framing bias
        eye_bonus     : 2 eyes → 1.6×, 1 eye → 1.0×, 0 eyes → 0.4×
                        (soft penalty — not a hard gate)
        sharpness²    : (log(1+lap_var) / log(1+1000))²
                        Squared so in-focus foreground beats blurry background
                        even when area and position are similar.  Log compresses
                        the wide variance range (50 → 5000+).
        """
        img_area = float(img_h * img_w)
        cx_img = img_w / 2.0
        cy_target = img_h * 0.35  # rule of thirds
        sigma = 0.4 * np.sqrt(img_w**2 + img_h**2)
        sharp_norm = np.log1p(1000.0)
        eye_bonus = {0: 0.4, 1: 1.0, 2: 1.6}

        def _score(f):
            x, y, w, h, n_eyes, lap_var = f
            area = (w * h) / img_area
            dist_sq = (x + w / 2 - cx_img) ** 2 + (y + h / 2 - cy_target) ** 2
            cw = np.exp(-dist_sq / (2 * sigma**2))
            eb = eye_bonus.get(min(n_eyes, 2), 1.6)
            sharpness = (np.log1p(lap_var) / sharp_norm) ** 2
            return area * cw * eb * sharpness

        best = max(faces, key=_score)
        return best[:4]  # (x, y, w, h)

    # ------------------------------------------------------------------
    # Per-row preprocessing
    # ------------------------------------------------------------------

    def preprocess_single(self, data_row) -> np.ndarray:
        img = data_row["img"]
        faces = self.detect_faces(img)
        if not faces:
            return _center_crop_fallback(img, self.face_size)
        x, y, w, h = self._best_face(faces, img.shape[0], img.shape[1])
        return cv2.resize(
            img[y : y + h, x : x + w], self.face_size, interpolation=cv2.INTER_AREA
        ).astype(np.float32)

    def _process_row(self, args):
        idx, row = args
        return idx, self.preprocess_single(row)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def __call__(self, data, multi_face: bool = False) -> np.ndarray:
        """
        Parameters
        ----------
        data       : DataFrame with an 'img' column
        multi_face : reserved for future use (currently ignored)

        Returns
        -------
        np.ndarray of shape (N, H, W, 3) float32
        """
        rows = list(data.iterrows())
        results = [None] * len(rows)

        with ThreadPoolExecutor(max_workers=self.n_jobs) as pool:
            futures = {
                pool.submit(self._process_row, (i, row)): i
                for i, (_, row) in enumerate(rows)
            }
            for future in as_completed(futures):
                i, face = future.result()
                results[i] = face

        return np.stack(results).astype(np.float32)


# =============================================================================
# DNN Face Detector
# =============================================================================


class DNNFaceDetector:
    """
    OpenCV ResNet-SSD face detector with sharpness-aware candidate scoring.

    Why DNN over Haar
    -----------------
    The Haar pipeline required 7 hand-tuned filters to suppress false positives
    (shirts, walls, cartoons, background faces).  The DNN model learned to
    reject these internally during training, so almost no post-filtering is
    needed.  Confidence scores are also more semantically meaningful than
    Haar's minNeighbors heuristic.

    Scoring
    -------
    When multiple faces are detected, the best is chosen by:

        score = confidence² × area × center_weight × sharpness

    confidence²   : squared so high-confidence detections strongly dominate
    area          : fraction of image covered
    center_weight : Gaussian decay from rule-of-thirds target (35 % from top)
    sharpness     : log(1 + Laplacian_variance) / log(1 + 1000)
                    Key tiebreaker for group shots where conf and area are
                    similar — the in-focus foreground subject is always sharper.

    Parameters
    ----------
    path                 : directory to cache model files
    face_size            : output crop size as (width, height) tuple
    confidence_threshold : minimum DNN confidence to consider a detection
                           (default 0.5; raise to 0.7 to reduce false positives,
                           lower to 0.3 if valid faces are being missed)
    """

    _PROTO_URL = (
        "https://raw.githubusercontent.com/opencv/opencv/master/samples/"
        "dnn/face_detector/deploy.prototxt"
    )
    _MODEL_URL = (
        "https://github.com/opencv/opencv_3rdparty/raw/"
        "dnn_samples_face_detector_20170830/"
        "res10_300x300_ssd_iter_140000.caffemodel"
    )

    def __init__(
        self,
        path: str,
        face_size: tuple,
        confidence_threshold: float = 0.5,
    ) -> None:
        self.face_size = face_size
        self.confidence_threshold = confidence_threshold

        os.makedirs(path, exist_ok=True)

        self._prototxt = os.path.join(path, "deploy.prototxt")
        self._caffemodel = os.path.join(
            path, "res10_300x300_ssd_iter_140000.caffemodel"
        )

        if not os.path.exists(self._prototxt):
            _download_file(self._PROTO_URL, self._prototxt)
        if not os.path.exists(self._caffemodel):
            _download_file(self._MODEL_URL, self._caffemodel)

        # Thread-local storage — cv2.dnn nets are not thread-safe when shared.
        self._tls = threading.local()

    # ------------------------------------------------------------------
    # Thread-local net access
    # ------------------------------------------------------------------

    def _get_net(self):
        if not hasattr(self._tls, "net"):
            self._tls.net = cv2.dnn.readNetFromCaffe(self._prototxt, self._caffemodel)
        return self._tls.net

    # ------------------------------------------------------------------
    # Sharpness
    # ------------------------------------------------------------------

    @staticmethod
    def _sharpness(img: np.ndarray, x: int, y: int, w: int, h: int) -> float:
        """
        Laplacian variance of the grayscale face patch, log-compressed.

        Uses the RAW (non-equalized) image so the score reflects true
        optical focus rather than artificially boosted contrast.
        log1p compresses the wide range (50 – 5000+) so extreme values
        don't dominate the composite score.
        """
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        patch = gray[y : y + h, x : x + w]
        lap_var = cv2.Laplacian(patch, cv2.CV_64F).var()
        return float(np.log1p(lap_var))

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------

    def detect_faces(self, img: np.ndarray):
        """
        Runs the ResNet-SSD detector on *img* and returns all detections
        above *confidence_threshold* as a list of
        (x, y, w, h, confidence, sharpness) tuples sorted by confidence
        descending.
        """
        net = self._get_net()
        h, w = img.shape[:2]

        blob = cv2.dnn.blobFromImage(
            cv2.resize(cv2.cvtColor(img, cv2.COLOR_RGB2BGR), (300, 300)),
            1.0,
            (300, 300),
            (104.0, 177.0, 123.0),
        )
        net.setInput(blob)
        dets = net.forward()

        faces = []
        for i in range(dets.shape[2]):
            conf = float(dets[0, 0, i, 2])
            if conf < self.confidence_threshold:
                continue

            x1 = max(0, int(dets[0, 0, i, 3] * w))
            y1 = max(0, int(dets[0, 0, i, 4] * h))
            x2 = min(w, int(dets[0, 0, i, 5] * w))
            y2 = min(h, int(dets[0, 0, i, 6] * h))
            fw, fh = x2 - x1, y2 - y1

            if fw > 0 and fh > 0:
                sharp = self._sharpness(img, x1, y1, fw, fh)
                faces.append((x1, y1, fw, fh, conf, sharp))

        return sorted(faces, key=lambda f: f[4], reverse=True)

    # ------------------------------------------------------------------
    # Face selection
    # ------------------------------------------------------------------

    def _best_face(self, faces, img_h: int, img_w: int):
        """
        Returns the (x, y, w, h) of the highest-scoring candidate.

        Score = confidence² × area × center_weight × sharpness_weight
        """
        img_area = float(img_h * img_w)
        cx_img = img_w / 2.0
        cy_target = img_h * 0.35
        sigma = 0.4 * np.sqrt(img_w**2 + img_h**2)
        sharp_norm = np.log1p(1000.0)

        def _score(f):
            x, y, w, h, conf, sharp = f
            area = (w * h) / img_area
            dist_sq = (x + w / 2 - cx_img) ** 2 + (y + h / 2 - cy_target) ** 2
            cw = np.exp(-dist_sq / (2 * sigma**2))
            sw = sharp / sharp_norm
            return (conf**2) * area * cw * sw

        best = max(faces, key=_score)
        return best[:4]  # (x, y, w, h)

    # ------------------------------------------------------------------
    # Per-row preprocessing
    # ------------------------------------------------------------------

    def preprocess_single(self, data_row) -> np.ndarray:
        img = data_row["img"]
        faces = self.detect_faces(img)
        if not faces:
            return _center_crop_fallback(img, self.face_size)
        x, y, w, h = self._best_face(faces, img.shape[0], img.shape[1])
        return cv2.resize(
            img[y : y + h, x : x + w], self.face_size, interpolation=cv2.INTER_AREA
        ).astype(np.float32)

    def _process_row(self, args):
        idx, row = args
        return idx, self.preprocess_single(row)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def __call__(self, data) -> np.ndarray:
        """
        Parameters
        ----------
        data : DataFrame with an 'img' column

        Returns
        -------
        np.ndarray of shape (N, H, W, 3) float32
        """
        rows = list(data.iterrows())
        results = [None] * len(rows)
        n_jobs = min(os.cpu_count() or 4, 8)

        with ThreadPoolExecutor(max_workers=n_jobs) as pool:
            futures = {
                pool.submit(self._process_row, (i, row)): i
                for i, (_, row) in enumerate(rows)
            }
            for future in as_completed(futures):
                i, face = future.result()
                results[i] = face

        return np.stack(results).astype(np.float32)


# =============================================================================
# Convenience runner — called when the file is run directly
# =============================================================================


def run_preprocessing(
    train,
    test,
    detector: str = "dnn",
    face_size: tuple = (100, 100),
    model_path: str = "tmp",
    cache_path: str | None = None,
):
    """
    Runs the chosen detector on *train* and *test* DataFrames.

    Parameters
    ----------
    train      : training DataFrame with 'img' column
    test       : test DataFrame with 'img' column
    detector   : 'dnn' (recommended) or 'haar'
    face_size  : output crop dimensions (width, height)
    model_path : where to cache detector model files
    cache_path : if given, saves/loads preprocessed arrays from this directory
                 to skip reprocessing on subsequent runs

    Returns
    -------
    train_X : np.ndarray  (N_train, H, W, 3)
    train_y : np.ndarray  (N_train,)
    test_X  : np.ndarray  (N_test,  H, W, 3)
    """
    import shutil

    # --- Load from cache if available ---
    if cache_path and os.path.exists(cache_path):
        print(f"Loading cached arrays from '{cache_path}' ...")
        train_X = np.load(os.path.join(cache_path, "train_X.npy"))
        train_y = np.load(os.path.join(cache_path, "train_y.npy"))
        test_X = np.load(os.path.join(cache_path, "test_X.npy"))
        print(
            f"  train_X: {train_X.shape}  train_y: {train_y.shape}  test_X: {test_X.shape}"
        )
        return train_X, train_y, test_X

    # --- Choose detector ---
    if detector == "dnn":
        print("Using DNNFaceDetector ...")
        preprocessor = DNNFaceDetector(path=model_path, face_size=face_size)
    elif detector == "haar":
        print("Using HAARPreprocessor ...")
        preprocessor = HAARPreprocessor(path=model_path, face_size=face_size)
    else:
        raise ValueError(f"Unknown detector '{detector}'. Choose 'dnn' or 'haar'.")

    # --- Run preprocessing ---
    print("Preprocessing training set ...")
    train_X = preprocessor(train)
    train_y = train["class"].values

    print("Preprocessing test set ...")
    test_X = preprocessor(test)

    print(
        f"  train_X: {train_X.shape}  train_y: {train_y.shape}  test_X: {test_X.shape}"
    )

    # --- Optionally cache results ---
    if cache_path:
        if os.path.exists(cache_path):
            shutil.rmtree(cache_path)
        os.makedirs(cache_path)
        np.save(os.path.join(cache_path, "train_X.npy"), train_X)
        np.save(os.path.join(cache_path, "train_y.npy"), train_y)
        np.save(os.path.join(cache_path, "test_X.npy"), test_X)
        print(f"  Cached to '{cache_path}'")

    return train_X, train_y, test_X
