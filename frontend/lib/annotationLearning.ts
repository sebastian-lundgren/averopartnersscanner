export type BboxLike = { x: number; y: number; w: number; h: number };

export function parseBbox(b: unknown): BboxLike | null {
  if (!b || typeof b !== "object") return null;
  const o = b as Record<string, unknown>;
  if (Array.isArray(o.boxes) && o.boxes.length > 0) {
    return parseBbox(o.boxes[0]);
  }
  if (typeof o.x !== "number" || typeof o.y !== "number") return null;
  const w = typeof o.w === "number" ? o.w : 0;
  const h = typeof o.h === "number" ? o.h : 0;
  if (w < 1e-6 || h < 1e-6) return null;
  return { x: o.x, y: o.y, w, h };
}

export function bboxIou(a: BboxLike, b: BboxLike): number {
  const ax2 = a.x + a.w;
  const ay2 = a.y + a.h;
  const bx2 = b.x + b.w;
  const by2 = b.y + b.h;
  const ix = Math.max(0, Math.min(ax2, bx2) - Math.max(a.x, b.x));
  const iy = Math.max(0, Math.min(ay2, by2) - Math.max(a.y, b.y));
  const inter = ix * iy;
  const u = a.w * a.h + b.w * b.h - inter;
  return u <= 0 ? 0 : inter / u;
}

export type RowForSummary = {
  original_model_status: string | null;
  model_predicted_status: string | null;
  training_label: string | null;
  final_status: string;
  error_type: string | null;
  comment: string | null;
  manual_bbox: Record<string, number> | null;
  model_bbox: unknown;
};

export type CropSource = "manual" | "model" | "none";

/**
 * Korte, regelbaserte linjer utelukkende fra lagrede felt og enkle utregninger (f.eks. IoU).
 * Ikke en «forklaring» fra modellen.
 */
export function ruleBasedAnnotationSummaryLines(
  r: RowForSummary,
  cropSource: CropSource
): string[] {
  const manual = parseBbox(r.manual_bbox);
  const model = parseBbox(r.model_bbox);

  const lines: string[] = [
    "Regelbasert oppsummering fra appen (ikke tekst eller kommentar fra YOLO/modellen).",
  ];

  lines.push(
    `Utsnitt i «Utsnitt»-kolonnen: ${
      cropSource === "manual"
        ? "manuell bbox (normaliserte koordinater lagret i raden)"
        : cropSource === "model"
          ? "modell bbox som reserve (ingen manuell bbox lagret)"
          : "ingen bbox — tomt utsnitt"
    }.`
  );

  lines.push(`Treningslabel (lagret): ${r.training_label ?? "—"}`);
  lines.push(`Feiltype (lagret): ${r.error_type ?? "—"}`);
  lines.push(
    r.comment?.trim()
      ? `Notat (lagret): «${r.comment.trim()}»`
      : "Notat (lagret): —"
  );

  lines.push(`Modellstatus opprinnelig (lagret): ${r.original_model_status ?? "—"}`);
  if (r.model_predicted_status && r.model_predicted_status !== r.original_model_status) {
    lines.push(`Prediksjonsstatus (koblet prediksjon): ${r.model_predicted_status}`);
  }
  lines.push(`Human status lagret (review): ${r.final_status}`);

  lines.push(`Manuell bbox: ${manual ? "lagret (se egen kolonne)" : "ikke lagret"}`);
  lines.push(`Modell bbox: ${model ? "fra prediksjon (se egen kolonne)" : "ingen"}`);

  if (manual && model) {
    lines.push(`IoU mellom manuell og modell-bbox (beregnet i appen): ${bboxIou(manual, model).toFixed(3)}`);
  } else {
    lines.push("IoU mellom manuell og modell-bbox: — (trenger begge)");
  }

  return lines;
}
