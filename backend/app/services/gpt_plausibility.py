"""
OpenAI vision: plausibilitets-gate for alarmskilt (scene-forståelse, ikke ren detektor).
Returnerer strukturert JSON brukt av prediction-routing.
"""

from __future__ import annotations

import base64
import io
import json
import os
import platform as plat
import re
from dataclasses import dataclass, field
from typing import Any

from PIL import Image

from app.config import settings


@dataclass
class GptDinoCandidateResult:
    """GPT vurdering av et eksisterende DINO-treff (etter detektor, ikke portvakt)."""

    plausibility_score: int = 0
    dino_box_plausible: bool = False
    guardrail_passed: bool = False
    reject_reasons: list[str] = field(default_factory=list)
    rationale_short: str = ""
    api_failed: bool = False
    raw_error: str | None = None

    def accepts_dino_box(self) -> bool:
        if self.api_failed:
            return True
        return bool(self.dino_box_plausible and self.guardrail_passed)

    @classmethod
    def from_parsed(cls, d: dict[str, Any]) -> GptDinoCandidateResult:
        score = d.get("plausibility_score", 0)
        try:
            score = int(round(float(score)))
        except (TypeError, ValueError):
            score = 0
        score = max(0, min(100, score))
        reasons = d.get("reject_reasons") or []
        if not isinstance(reasons, list):
            reasons = []
        reasons = [str(x) for x in reasons]
        plausible = bool(d.get("dino_box_plausible", False))
        return cls(
            plausibility_score=score,
            dino_box_plausible=plausible,
            guardrail_passed=bool(d.get("guardrail_passed", False)),
            reject_reasons=reasons,
            rationale_short=str(d.get("rationale_short", "")).strip() or "(no rationale_short)",
        )


@dataclass
class GptPlausibilityResult:
    plausibility_score: int = 0
    likely_alarm_sign: bool = False
    route: str = "unclear"  # direct_positive | send_to_dino | unclear
    guardrail_passed: bool = False
    reject_reasons: list[str] = field(default_factory=list)
    rationale_short: str = ""
    likely_region_hint: str | None = None
    image_quality_flags: list[str] | None = None
    api_failed: bool = False
    raw_error: str | None = None

    @classmethod
    def from_parsed(cls, d: dict[str, Any]) -> GptPlausibilityResult:
        score = d.get("plausibility_score", 0)
        try:
            score = int(round(float(score)))
        except (TypeError, ValueError):
            score = 0
        score = max(0, min(100, score))
        route = str(d.get("route", "unclear")).strip().lower()
        if route not in ("direct_positive", "send_to_dino", "unclear"):
            route = "unclear"
        flags = d.get("image_quality_flags")
        if flags is not None and not isinstance(flags, list):
            flags = None
        reasons = d.get("reject_reasons") or []
        if not isinstance(reasons, list):
            reasons = []
        reasons = [str(x) for x in reasons]
        hint = d.get("likely_region_hint")
        hint = str(hint).strip() if hint else None
        return cls(
            plausibility_score=score,
            likely_alarm_sign=bool(d.get("likely_alarm_sign", False)),
            route=route,
            guardrail_passed=bool(d.get("guardrail_passed", False)),
            reject_reasons=reasons,
            rationale_short=str(d.get("rationale_short", "")).strip() or "(no rationale_short)",
            likely_region_hint=hint,
            image_quality_flags=flags,
        )


def _pil_to_jpeg_b64(im: Image.Image, max_side: int = 1536) -> str:
    w, h = im.size
    if max(w, h) > max_side:
        scale = max_side / float(max(w, h))
        im = im.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    # Dropp ICC/exif-metadata som kan inneholde tekst som noen stakker proever som latin-1/ascii.
    save_kw: dict = {"format": "JPEG", "quality": 88}
    if hasattr(Image, "Exif"):
        save_kw["exif"] = Image.Exif()
    im.save(buf, **save_kw)
    return base64.standard_b64encode(buf.getvalue()).decode("ascii")


def _parse_json_object(content: str) -> dict[str, Any]:
    text = content.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```\s*$", "", text)
    return json.loads(text)


# Kun ASCII her: unngar at mellomledd (logging, eldre HTTP-stakk) proever ascii-koding paa payload.
_SYSTEM_PROMPT = """You are a strict visual plausibility reviewer for burglar / security alarm signs on building facades in screenshots (e.g. Street View style). You are NOT a bounding-box detector.

Return a single JSON object with these keys (exact names):
- plausibility_score: integer 0-100 (how plausible that a real alarm/security sign is present and relevant)
- likely_alarm_sign: boolean
- route: one of "direct_positive", "send_to_dino", "unclear"
- guardrail_passed: boolean - true only if a positive would be on facade/door/glass near entrance, NOT on UI overlay, vegetation, sky, image edge, house number only, random box, or large UI chrome
- reject_reasons: array of short strings (empty if none) e.g. "ui_overlay", "vegetation", "house_number", "sky", "edge", "reflection", "blur", "too_far"
- rationale_short: one short sentence (Norwegian or English, UTF-8 safe text only)
- likely_region_hint: optional string, coarse region if useful (e.g. "door_right") or null
- image_quality_flags: optional array of strings or null

Rules:
- Favor small sign-like objects on facade, door frame, or glass by entrance.
- Penalize Google Maps/Street View UI, bushes, sky, frame edges, door numbers as primary finding, generic labels, large UI panels.
- route "direct_positive" only if you would be very confident AND placement is plausible (you still output guardrail_passed accordingly).
- route "send_to_dino" if an alarm sign might exist but you are not sure enough for a direct call, or geometry needs a detector.
- route "unclear" if there is no plausible alarm-sign candidate or image is too poor.

Respond with JSON only, no markdown."""

_SYSTEM_PROMPT_DINO_BOX = """You validate a SINGLE bounding box from an open-vocabulary detector (Grounding DINO) on a street-view style image (UI already cropped from edges).

The detector claims an alarm/security sign may appear inside the given box (normalized x,y,w,h each 0-1 relative to this image). Your job is plausibility only: is that region actually a plausible burglar alarm sign on the building, or a false hit?

Return one JSON object with exact keys:
- plausibility_score: integer 0-100
- dino_box_plausible: boolean — true only if the boxed region could reasonably be an alarm sign on facade, door, or entrance glass
- guardrail_passed: boolean — true only if the box is NOT primarily: Maps/Street View UI overlay, vegetation, sky, frame edge artifact, house number plate only, random clutter, or clearly wrong object
- reject_reasons: array of short strings (empty if none), e.g. "ui_overlay", "vegetation", "house_number", "sky", "edge", "random_box"
- rationale_short: one short sentence (Norwegian or English, ASCII-safe where possible)

Favor signs near entrance, on facade, or in door glass. Penalize UI chrome, bushes, sky strips, image border, numbers-only plaques, and spurious boxes.

Respond with JSON only, no markdown."""


def _optional_ascii_env(var: str) -> str | None:
    """HTTP-headere via h11 maa vaere ASCII; dropp verdier med non-ASCII (f.eks. feil i OPENAI_ORG_ID)."""
    v = os.environ.get(var)
    if v is None:
        return None
    v = v.strip()
    if not v:
        return None
    try:
        v.encode("ascii")
    except UnicodeEncodeError:
        return None
    return v


def _safe_stainless_headers() -> dict[str, str]:
    """Overstyr OpenAI-SDK sine X-Stainless-* headere (unngaa platform.platform() m. lokale tegn)."""
    try:
        from openai import __version__ as openai_ver
    except Exception:
        openai_ver = "0"
    os_name = "Unknown"
    try:
        s = plat.system().lower()
        if s == "darwin":
            os_name = "MacOS"
        elif s == "windows":
            os_name = "Windows"
        elif s == "linux":
            os_name = "Linux"
    except Exception:
        pass
    arch = "unknown"
    try:
        m = plat.machine().lower()
        if m in ("arm64", "aarch64"):
            arch = "arm64"
        elif m in ("x86_64", "amd64", "x64"):
            arch = "x64"
        elif m == "arm":
            arch = "arm"
    except Exception:
        pass
    return {
        "X-Stainless-Lang": "python",
        "X-Stainless-Package-Version": openai_ver,
        "X-Stainless-OS": os_name,
        "X-Stainless-Arch": arch,
        "X-Stainless-Runtime": "CPython",
        "X-Stainless-Runtime-Version": plat.python_version(),
    }


def _safe_raw_error(exc: BaseException) -> str:
    """Avoid secondary UnicodeEncodeError when error strings hit ASCII-only handlers."""
    if isinstance(exc, UnicodeEncodeError):
        obj = exc.object
        if isinstance(obj, (bytes, bytearray)):
            return (
                f"UnicodeEncodeError: codec={exc.encoding!r} reason={exc.reason!r} "
                f"byte_range={exc.start}-{exc.end}"
            )
        return (
            f"UnicodeEncodeError: codec={exc.encoding!r} at char index {exc.start}-{exc.end}"
        )
    try:
        return str(exc)[:500]
    except Exception:
        return f"{type(exc).__name__}"


def run_plausibility_gate(infer_image: Image.Image) -> GptPlausibilityResult:
    key = (settings.openai_api_key or "").strip()
    if not key:
        return GptPlausibilityResult(api_failed=True, raw_error="missing_api_key")

    model = (settings.openai_vision_model or "gpt-4o-mini").strip()
    try:
        from openai import OpenAI

        # organization=None laater SDK lese OPENAI_ORG_ID paa nytt (kan inneholde non-ASCII).
        org = _optional_ascii_env("OPENAI_ORG_ID")
        proj = _optional_ascii_env("OPENAI_PROJECT_ID")
        client = OpenAI(
            api_key=key,
            timeout=120.0,
            organization=org if org is not None else "",
            project=proj if proj is not None else "",
            default_headers=_safe_stainless_headers(),
        )
        b64 = _pil_to_jpeg_b64(infer_image.convert("RGB"))
        user_text = (
            "Analyze this image (already cropped to remove typical top/left/right/bottom browser UI). "
            "Apply the rules and output the JSON object."
        )
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_text},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                        },
                    ],
                },
            ],
            response_format={"type": "json_object"},
            max_tokens=500,
        )
        content = (resp.choices[0].message.content or "").strip()
        parsed = _parse_json_object(content)
        return GptPlausibilityResult.from_parsed(parsed)
    except Exception as e:
        return GptPlausibilityResult(
            api_failed=True,
            raw_error=_safe_raw_error(e),
            rationale_short="GPT call failed",
        )


def run_dino_candidate_plausibility(
    infer_image: Image.Image,
    *,
    bbox_norm: dict[str, float],
    dino_label: str,
    dino_score: float,
) -> GptDinoCandidateResult:
    """
    Etter DINO: vurder om den foreslåtte boksen er plausibel alarmskilt i scenen.
    Uten API-nøkkel: api_failed=True slik at caller beholder ren DINO-beslutning.
    """
    key = (settings.openai_api_key or "").strip()
    if not key:
        return GptDinoCandidateResult(api_failed=True, raw_error="missing_api_key")

    model = (settings.openai_vision_model or "gpt-4o-mini").strip()
    bx = bbox_norm.get("x", 0.0)
    by = bbox_norm.get("y", 0.0)
    bw = bbox_norm.get("w", 0.0)
    bh = bbox_norm.get("h", 0.0)
    user_text = (
        "Grounding DINO proposed one candidate box on this image.\n"
        f"- Normalized box (x,y,w,h in 0..1 relative to this image): x={bx:.4f}, y={by:.4f}, w={bw:.4f}, h={bh:.4f}\n"
        f"- Detector label text: {dino_label!r}\n"
        f"- Detector confidence (model score, 0-1): {dino_score:.3f}\n"
        "Validate whether this box plausibly contains a burglar/security alarm sign on the building "
        "(not UI, vegetation, sky, edge, house number only, or random false positive). Output the JSON object."
    )
    try:
        from openai import OpenAI

        org = _optional_ascii_env("OPENAI_ORG_ID")
        proj = _optional_ascii_env("OPENAI_PROJECT_ID")
        client = OpenAI(
            api_key=key,
            timeout=120.0,
            organization=org if org is not None else "",
            project=proj if proj is not None else "",
            default_headers=_safe_stainless_headers(),
        )
        b64 = _pil_to_jpeg_b64(infer_image.convert("RGB"))
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT_DINO_BOX},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_text},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                        },
                    ],
                },
            ],
            response_format={"type": "json_object"},
            max_tokens=500,
        )
        content = (resp.choices[0].message.content or "").strip()
        parsed = _parse_json_object(content)
        return GptDinoCandidateResult.from_parsed(parsed)
    except Exception as e:
        return GptDinoCandidateResult(
            api_failed=True,
            raw_error=_safe_raw_error(e),
            rationale_short="GPT DINO-validation call failed",
        )
