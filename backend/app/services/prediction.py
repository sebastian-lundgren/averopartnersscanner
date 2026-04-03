"""
Prediksjon: hard pre-crop → Grounding DINO Base (HF) som ren bbox-kandidatfinner.
Automatisk skilt_funnet brukes ikke; status er uklart/trenger_manuell + ev. bbox-forslag for manuell merking.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

import cv2
from PIL import Image

from app.config import settings
from app.models import ReviewStatus
from app.services.quality import assess_image_quality

# Deterministisk Street View / Maps pre-crop (andel av bredde/høyde kuttet fra hver kant).
_SV_TOP_FRAC = 0.20
_SV_LEFT_FRAC = 0.26
_SV_RIGHT_FRAC = 0.07
_SV_BOTTOM_FRAC = 0.08


@dataclass
class PredictionResult:
    status: ReviewStatus
    confidence: int
    bbox_norm: dict | None  # x,y,w,h 0–1
    rationale: str


@dataclass
class _DinoHit:
    label_txt: str
    best_score: float
    bbox_norm: dict


_lock = threading.Lock()
_model = None
_processor = None
_device = None


def _pick_device():
    import torch

    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _load_dino():
    global _model, _processor, _device
    with _lock:
        if _model is None:
            from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

            mid = settings.grounding_dino_model_id
            _processor = AutoProcessor.from_pretrained(mid)
            _model = AutoModelForZeroShotObjectDetection.from_pretrained(mid)
            _device = _pick_device()
            _model.to(_device)
            _model.eval()
    return _model, _processor, _device


def _phrase_list() -> list[str]:
    parts = [p.strip() for p in settings.grounding_dino_phrases.split(",") if p.strip()]
    return parts if parts else ["alarm sign", "alarm sticker"]


def _streetview_precrop_box(iw: int, ih: int) -> tuple[int, int, int, int]:
    """Hard pre-crop: ingen utvidelse tilbake til full ramme."""
    x0 = int(iw * _SV_LEFT_FRAC)
    y0 = int(ih * _SV_TOP_FRAC)
    x1 = int(iw * (1.0 - _SV_RIGHT_FRAC))
    y1 = int(ih * (1.0 - _SV_BOTTOM_FRAC))
    x0 = max(0, min(x0, max(0, iw - 2)))
    y0 = max(0, min(y0, max(0, ih - 2)))
    x1 = max(x0 + 2, min(max(x0 + 2, x1), iw))
    y1 = max(y0 + 2, min(max(y0 + 2, y1), ih))
    return x0, y0, x1, y1


def _xyxy_to_norm_xywh(
    box,
    img_w: int,
    img_h: int,
    *,
    x_off: float = 0.0,
    y_off: float = 0.0,
) -> dict:
    t = box.tolist() if hasattr(box, "tolist") else list(box)
    x1, y1, x2, y2 = float(t[0]), float(t[1]), float(t[2]), float(t[3])
    x1, y1 = x1 + x_off, y1 + y_off
    x2, y2 = x2 + x_off, y2 + y_off
    x1, y1 = max(0.0, x1), max(0.0, y1)
    x2, y2 = min(float(img_w), x2), min(float(img_h), y2)
    w, h = max(0.0, x2 - x1), max(0.0, y2 - y1)
    return {
        "x": x1 / img_w,
        "y": y1 / img_h,
        "w": w / img_w,
        "h": h / img_h,
    }


def _dino_infer_or_terminal(
    infer_image: Image.Image,
    iw: int,
    ih: int,
    rx0: int,
    ry0: int,
    q,
    q_note: str,
) -> PredictionResult | _DinoHit:
    try:
        model, processor, device = _load_dino()
    except Exception as e:
        return PredictionResult(
            status=ReviewStatus.TRENGER_MANUELL,
            confidence=20,
            bbox_norm=None,
            rationale=f"Grounding DINO kunne ikke lastes ({e!s}). Installer torch+transformers og prøv igjen.",
        )

    import torch

    phrases = _phrase_list()
    text_labels = [phrases]

    try:
        inputs = processor(images=infer_image, text=text_labels, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        results = processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=settings.grounding_dino_box_threshold,
            text_threshold=settings.grounding_dino_text_threshold,
            target_sizes=[infer_image.size[::-1]],
        )
    except Exception as e:
        return PredictionResult(
            status=ReviewStatus.TRENGER_MANUELL,
            confidence=25,
            bbox_norm=None,
            rationale=f"Grounding DINO-inferens feilet ({e!s}).{q_note}",
        )

    res0 = results[0]
    boxes = res0.get("boxes")
    scores = res0.get("scores")
    labels = res0.get("labels")
    n_det = int(boxes.shape[0]) if boxes is not None and hasattr(boxes, "shape") else 0
    if boxes is None or n_det == 0:
        return PredictionResult(
            status=ReviewStatus.TRENGER_MANUELL,
            confidence=int(22 + min(25, q.combined * 40)),
            bbox_norm=None,
            rationale="Grounding DINO: ingen treff over terskel for alarm/skilt-relaterte fraser — unknown-first."
            + q_note,
        )

    best_i = int(torch.argmax(scores).item())
    best_score = float(scores[best_i].item())
    best_box = boxes[best_i]
    label_txt = "?"
    if labels is not None:
        label_txt = labels[best_i]
        if hasattr(label_txt, "item"):
            label_txt = label_txt.item()
        label_txt = str(label_txt)
    bbox_norm = _xyxy_to_norm_xywh(best_box, iw, ih, x_off=float(rx0), y_off=float(ry0))
    return _DinoHit(label_txt=label_txt, best_score=best_score, bbox_norm=bbox_norm)


def _hit_to_prediction(hit: _DinoHit, q, q_note: str) -> PredictionResult:
    label_txt = hit.label_txt
    best_score = hit.best_score
    bbox_norm = hit.bbox_norm

    if q.combined < 0.26:
        conf = int(min(72, max(30, round(best_score * 85))))
        return PredictionResult(
            status=ReviewStatus.UKLART,
            confidence=conf,
            bbox_norm=bbox_norm,
            rationale=(
                f"Grounding DINO: «{label_txt}» (score {best_score:.2f}), men bildekvalitet/sikt er svak — uklart."
                + q_note
            ),
        )

    if best_score >= 0.52:
        conf = int(min(94, max(58, round(best_score * 100))))
        return PredictionResult(
            status=ReviewStatus.UKLART,
            confidence=conf,
            bbox_norm=bbox_norm,
            rationale=(
                f"Grounding DINO-forslag «{label_txt}» (score {best_score:.2f}) — kun bbox-forslag, "
                f"automatisk skilt_funnet er av; merk manuelt i review (alarm_sign / not_alarm_sign / unclear)."
                + q_note
            ),
        )

    if best_score >= 0.32:
        conf = int(min(78, max(38, round(best_score * 100))))
        return PredictionResult(
            status=ReviewStatus.UKLART,
            confidence=conf,
            bbox_norm=bbox_norm,
            rationale=(
                f"Grounding DINO: svakt treff «{label_txt}» (score {best_score:.2f}) — bør menneske vurdere."
                + q_note
            ),
        )

    return PredictionResult(
        status=ReviewStatus.TRENGER_MANUELL,
        confidence=int(max(25, round(best_score * 70))),
        bbox_norm=bbox_norm,
        rationale=(
            f"Grounding DINO: svært lav score for «{label_txt}» ({best_score:.2f}) — trenger manuell vurdering."
            + q_note
        ),
    )


def _run_grounding_dino_on_precrop(
    infer_image: Image.Image,
    iw: int,
    ih: int,
    rx0: int,
    ry0: int,
    q,
    q_note: str,
) -> PredictionResult:
    out = _dino_infer_or_terminal(infer_image, iw, ih, rx0, ry0, q, q_note)
    if isinstance(out, PredictionResult):
        return out
    return _hit_to_prediction(out, q, q_note)


def run_heuristic_predict(image_path: str) -> PredictionResult:
    """
    Pre-crop → Grounding DINO (open-set kandidatfinner). Ingen GPT i prediksjonsflyten.
    """
    bgr = cv2.imread(image_path)
    if bgr is None:
        return PredictionResult(
            status=ReviewStatus.TRENGER_MANUELL,
            confidence=15,
            bbox_norm=None,
            rationale="Kunne ikke lese bildefil — trenger manuell vurdering.",
        )

    q = assess_image_quality(bgr)
    q_note = ""
    if q.flags:
        q_note = " Bildekvalitet: " + ", ".join(q.flags) + "."

    try:
        image = Image.open(image_path).convert("RGB")
    except OSError:
        return PredictionResult(
            status=ReviewStatus.TRENGER_MANUELL,
            confidence=15,
            bbox_norm=None,
            rationale="Kunne ikke åpne bilde for modell — trenger manuell vurdering.",
        )

    iw, ih = image.size
    rx0, ry0, rx1, ry1 = _streetview_precrop_box(iw, ih)
    infer_image = image.crop((rx0, ry0, rx1, ry1))
    return _run_grounding_dino_on_precrop(infer_image, iw, ih, rx0, ry0, q, q_note)


def map_confidence_to_review_hint(
    confidence: int,
    strong: int,
    unclear_hi: int,
    unclear_lo: int,
) -> str:
    if confidence >= strong:
        return "Sterk mistanke om skilt (krever likevel menneskelig QA ved lav terskel)"
    if unclear_lo <= confidence <= unclear_hi:
        return "Uklart / bør reviewes"
    return "Trenger manuell vurdering (lav score — ikke tolkes som 'ingen alarm')"
