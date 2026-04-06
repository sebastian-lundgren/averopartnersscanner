import type { BboxNorm } from "@/components/ReviewBboxEditor";

/** Aksepter tall fra JSON (noen lagrings-/serialiseringsveier gir strenger). */
function toNormBox(b: unknown): BboxNorm | null {
  if (!b || typeof b !== "object" || Array.isArray(b)) return null;
  const o = b as Record<string, unknown>;
  const x = Number(o.x);
  const y = Number(o.y);
  const w = Number(o.w);
  const h = Number(o.h);
  if (![x, y, w, h].every(Number.isFinite)) return null;
  if (w <= 1e-8 || h <= 1e-8) return null;
  return { x, y, w, h };
}

function isBbox(b: unknown): b is BboxNorm {
  return toNormBox(b) !== null;
}

/** YOLO multi-bbox metadata (valgfritt i bbox_json v2). */
export type YoloBboxTrustMeta = {
  yolo_trusted_primary?: boolean;
  yolo_primary_gate_reason?: string;
};

export function parseYoloTrustMeta(raw: unknown): YoloBboxTrustMeta {
  if (raw == null) return {};
  if (typeof raw === "string") {
    try {
      return parseYoloTrustMeta(JSON.parse(raw) as unknown);
    } catch {
      return {};
    }
  }
  if (typeof raw !== "object" || Array.isArray(raw)) return {};
  const o = raw as Record<string, unknown>;
  const trusted = o.yolo_trusted_primary;
  const reason = o.yolo_primary_gate_reason;
  return {
    yolo_trusted_primary: typeof trusted === "boolean" ? trusted : undefined,
    yolo_primary_gate_reason: typeof reason === "string" ? reason : undefined,
  };
}

/** True når backend eksplisitt sier at ingen pålitelig auto-primær (alle bokser er bare usikre forslag). */
export function yoloSuggestionsUncertain(raw: unknown): boolean {
  const m = parseYoloTrustMeta(raw);
  return m.yolo_trusted_primary === false;
}

/** Tolker prediction.bbox_json: legacy enkelt objekt, liste, eller { boxes, v }. */
export function parsePredBboxes(raw: unknown): BboxNorm[] {
  if (raw == null) return [];
  if (typeof raw === "string") {
    const t = raw.trim();
    if (!t) return [];
    try {
      return parsePredBboxes(JSON.parse(t) as unknown);
    } catch {
      return [];
    }
  }
  if (Array.isArray(raw)) {
    return raw.map(toNormBox).filter((b): b is BboxNorm => b !== null);
  }
  if (typeof raw === "object") {
    const o = raw as Record<string, unknown>;
    if (Array.isArray(o.boxes)) {
      return o.boxes.map(toNormBox).filter((b): b is BboxNorm => b !== null);
    }
    const one = toNormBox(o);
    if (one) return [one];
  }
  return [];
}
