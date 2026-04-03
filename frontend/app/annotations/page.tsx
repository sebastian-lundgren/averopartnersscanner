"use client";

import { useCallback, useEffect, useState } from "react";
import AnnotationCropPreview from "@/components/AnnotationCropPreview";
import { parseBbox, ruleBasedAnnotationSummaryLines } from "@/lib/annotationLearning";
import { API_BASE } from "@/lib/api";

type Row = {
  id: number;
  image_id: number;
  filename: string;
  original_model_status: string | null;
  model_predicted_status: string | null;
  model_bbox: Record<string, number> | null;
  manual_bbox: Record<string, number> | null;
  training_label: string | null;
  final_status: string;
  error_type: string | null;
  comment: string | null;
  dataset_split: string | null;
  created_at: string;
  annotated_by?: string | null;
};

type Overview = {
  rows: Row[];
  error_type_summary: [string, number][];
  total: number;
};

function fmtBbox(b: Record<string, number> | null) {
  if (!b || typeof b.x !== "number") return "—";
  return `${b.x.toFixed(3)},${b.y.toFixed(3)} ${b.w?.toFixed(3) ?? "?"}×${b.h?.toFixed(3) ?? "?"}`;
}

function previewForRow(r: Row): {
  bbox: ReturnType<typeof parseBbox>;
  source: "manual" | "model" | "none";
} {
  const manual = parseBbox(r.manual_bbox);
  if (manual) return { bbox: manual, source: "manual" };
  const model = parseBbox(r.model_bbox);
  if (model) return { bbox: model, source: "model" };
  return { bbox: null, source: "none" };
}

export default function AnnotationsPage() {
  const [data, setData] = useState<Overview | null>(null);
  const [msg, setMsg] = useState("");

  const load = useCallback(async () => {
    setMsg("");
    try {
      const r = await fetch(`${API_BASE}/api/training/annotations-overview?limit=500`);
      const j = await r.json();
      if (!r.ok) throw new Error(typeof j?.detail === "string" ? j.detail : JSON.stringify(j));
      setData(j as Overview);
    } catch (e) {
      setData(null);
      setMsg(e instanceof Error ? e.message : "Feil");
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  if (!data && !msg) return <p>Laster…</p>;
  if (!data) return <p className="muted">{msg || "Ingen data"}</p>;

  return (
    <>
      <h1>Annotering / læring</h1>
      <p className="muted">
        Oversikt over lagrede annoteringer: utsnitt, bbox-verdier, label, feiltype og notat. Kolonnen «Dataoppsummering»
        er automatisk tekst generert i appen fra disse feltene — ikke utsagn fra YOLO.
      </p>
      <p>
        <button type="button" className="secondary" onClick={() => void load()}>
          Oppdater
        </button>
      </p>

      {data.error_type_summary.length > 0 && (
        <div className="card" style={{ marginBottom: "1rem" }}>
          <h2 style={{ marginTop: 0, fontSize: "1rem" }}>Vanlige feiltyper (i dette utsnittet)</h2>
          <ul style={{ margin: 0, paddingLeft: "1.25rem" }}>
            {data.error_type_summary.map(([k, v]) => (
              <li key={k}>
                <code>{k}</code> — {v}
              </li>
            ))}
          </ul>
        </div>
      )}

      <p className="muted">{data.total} rader</p>
      <div style={{ overflowX: "auto" }}>
        <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 13 }}>
          <thead>
            <tr style={{ textAlign: "left", borderBottom: "1px solid var(--border)" }}>
              <th style={{ padding: "0.35rem" }}>Bilde</th>
              <th style={{ padding: "0.35rem" }}>Utsnitt</th>
              <th style={{ padding: "0.35rem", minWidth: 200 }}>Dataoppsummering (regelbasert i appen)</th>
              <th style={{ padding: "0.35rem" }}>Modell (oppr.)</th>
              <th style={{ padding: "0.35rem" }}>Modell bbox</th>
              <th style={{ padding: "0.35rem" }}>Manuell bbox</th>
              <th style={{ padding: "0.35rem" }}>Label</th>
              <th style={{ padding: "0.35rem" }}>Feiltype</th>
              <th style={{ padding: "0.35rem" }}>Notat</th>
              <th style={{ padding: "0.35rem" }}>Split</th>
              <th style={{ padding: "0.35rem" }}>Annotert av</th>
            </tr>
          </thead>
          <tbody>
            {data.rows.map((r) => {
              const prev = previewForRow(r);
              return (
              <tr key={r.id} style={{ borderBottom: "1px solid var(--border)" }}>
                <td style={{ padding: "0.35rem", whiteSpace: "nowrap" }}>
                  #{r.image_id}
                  <br />
                  <span className="muted">{r.filename}</span>
                </td>
                <td style={{ padding: "0.35rem", verticalAlign: "top" }}>
                  <AnnotationCropPreview imageId={r.image_id} bbox={prev.bbox} source={prev.source} />
                </td>
                <td style={{ padding: "0.35rem", verticalAlign: "top", maxWidth: 380, lineHeight: 1.45 }}>
                  <ul style={{ margin: 0, paddingLeft: "1.1rem" }}>
                    {ruleBasedAnnotationSummaryLines(r, prev.source).map((line, i) => (
                      <li key={i} style={{ marginBottom: "0.25rem" }}>
                        {line}
                      </li>
                    ))}
                  </ul>
                </td>
                <td style={{ padding: "0.35rem" }}>
                  {r.original_model_status ?? "—"}
                  {r.model_predicted_status && r.model_predicted_status !== r.original_model_status && (
                    <span className="muted">
                      <br />(pred: {r.model_predicted_status})
                    </span>
                  )}
                </td>
                <td style={{ padding: "0.35rem", fontFamily: "monospace", maxWidth: 140 }}>
                  {fmtBbox(r.model_bbox)}
                </td>
                <td style={{ padding: "0.35rem", fontFamily: "monospace", maxWidth: 140 }}>
                  {fmtBbox(r.manual_bbox)}
                </td>
                <td style={{ padding: "0.35rem" }}>{r.training_label ?? "—"}</td>
                <td style={{ padding: "0.35rem" }}>{r.error_type ?? "—"}</td>
                <td style={{ padding: "0.35rem", maxWidth: 200 }}>{r.comment ?? "—"}</td>
                <td style={{ padding: "0.35rem" }}>{r.dataset_split ?? "—"}</td>
                <td style={{ padding: "0.35rem", whiteSpace: "nowrap" }}>
                  {r.annotated_by ?? "—"}
                  <br />
                  <span className="muted" style={{ fontSize: 11 }}>
                    {r.created_at}
                  </span>
                </td>
              </tr>
            );
            })}
          </tbody>
        </table>
      </div>
    </>
  );
}
