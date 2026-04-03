"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import ReviewBboxEditor from "@/components/ReviewBboxEditor";
import type { BboxNorm } from "@/components/ReviewBboxEditor";
import { API_BASE, fileUrl } from "@/lib/api";
import {
  ANNOTATION_LABELS,
  AnnotationLabel,
  defaultAnnotationFromPredicted,
  ERROR_TYPES,
  reviewStatusForAnnotation,
} from "@/lib/constants";

type Pred = {
  id: number;
  predicted_status: string;
  confidence: number;
  rationale: string | null;
  bbox_json: BboxNorm | null;
  claimed_by?: string | null;
  claimed_at?: string | null;
};
type Img = { id: number; original_filename: string };
type QueueItem = { prediction: Pred; image: Img };

const ANNOTATOR_LS = "annotator_display_name";

export default function ReviewPage() {
  const [queue, setQueue] = useState<QueueItem[]>([]);
  const [idx, setIdx] = useState(0);
  const [annotationLabel, setAnnotationLabel] = useState<AnnotationLabel>("unclear");
  const [comment, setComment] = useState("");
  const [errorType, setErrorType] = useState("");
  const [msg, setMsg] = useState("");
  const [loading, setLoading] = useState(false);
  const [manualBbox, setManualBbox] = useState<BboxNorm | null>(null);
  const [yoloSplit, setYoloSplit] = useState<"" | "train" | "val" | "rejected">("train");
  const [annotatorId, setAnnotatorId] = useState("");
  const [queueStats, setQueueStats] = useState<{
    pending_review: number;
    free_or_expired_claim: number;
    claimed_active: number;
    completed_total: number;
  } | null>(null);

  useEffect(() => {
    try {
      const s = localStorage.getItem(ANNOTATOR_LS);
      if (s) setAnnotatorId(s);
    } catch {
      /* ignore */
    }
  }, []);

  useEffect(() => {
    try {
      if (annotatorId) localStorage.setItem(ANNOTATOR_LS, annotatorId);
    } catch {
      /* ignore */
    }
  }, [annotatorId]);

  const load = useCallback(async () => {
    setMsg("");
    try {
      const qs =
        annotatorId.trim() !== ""
          ? `&annotator_id=${encodeURIComponent(annotatorId.trim())}`
          : "";
      const [r, st] = await Promise.all([
        fetch(`${API_BASE}/api/reviews/queue?limit=100${qs}`),
        fetch(`${API_BASE}/api/reviews/queue-stats`),
      ]);
      const raw = await r.json();
      if (st.ok) {
        setQueueStats(
          (await st.json()) as {
            pending_review: number;
            free_or_expired_claim: number;
            claimed_active: number;
            completed_total: number;
          },
        );
      }
      if (!r.ok) {
        setQueue([]);
        setIdx(0);
        setMsg(typeof raw?.detail === "string" ? raw.detail : JSON.stringify(raw));
        return;
      }
      if (!Array.isArray(raw)) {
        setQueue([]);
        setIdx(0);
        setMsg("Uventet svar fra API (ikke en liste).");
        return;
      }
      const data = raw as QueueItem[];
      setQueue(data);
      setIdx(0);
      const firstStatus = data[0]?.prediction?.predicted_status;
      if (firstStatus) setAnnotationLabel(defaultAnnotationFromPredicted(firstStatus));
    } catch (e) {
      setQueue([]);
      setIdx(0);
      setMsg(e instanceof Error ? e.message : "Kunne ikke hente kø");
    }
  }, [annotatorId]);

  useEffect(() => {
    load();
  }, [load]);

  useEffect(() => {
    setIdx((i) => Math.min(i, Math.max(0, queue.length - 1)));
  }, [queue]);

  const current = queue.length ? queue[idx] : undefined;

  useEffect(() => {
    if (current) {
      setAnnotationLabel(defaultAnnotationFromPredicted(current.prediction.predicted_status));
      setManualBbox(null);
    }
  }, [current]);

  const submit = useCallback(
    async (approveOnly: boolean) => {
      const cur = queue[idx];
      if (!cur) return;
      setLoading(true);
      setMsg("");
      try {
        const body: Record<string, unknown> = {
          final_status: reviewStatusForAnnotation(annotationLabel),
          annotation_label: annotationLabel,
          comment: comment || null,
          error_type: errorType || null,
          approve_without_change: approveOnly,
        };
        if (manualBbox) body.annotation_bbox_json = manualBbox;
        if (yoloSplit) body.yolo_dataset_split = yoloSplit;
        const headers: Record<string, string> = { "Content-Type": "application/json" };
        const aid = annotatorId.trim();
        if (aid) headers["X-Annotator-Id"] = aid;
        if (aid) body.annotator_id = aid;
        const r = await fetch(`${API_BASE}/api/reviews/${cur.prediction.id}/submit`, {
          method: "POST",
          headers,
          body: JSON.stringify(body),
        });
        if (!r.ok) throw new Error(await r.text());
        setComment("");
        setErrorType("");
        const rest = queue.filter((_, i) => i !== idx);
        setQueue(rest);
        setIdx(0);
        if (rest[0]) setAnnotationLabel(defaultAnnotationFromPredicted(rest[0].prediction.predicted_status));
        setMsg("Lagret");
      } catch (e) {
        setMsg(e instanceof Error ? e.message : "Feil");
      } finally {
        setLoading(false);
      }
    },
    [queue, idx, annotationLabel, manualBbox, yoloSplit, comment, errorType, annotatorId]
  );

  const claimNext = useCallback(async () => {
    const aid = annotatorId.trim();
    if (!aid) {
      setMsg("Skriv inn navn under «Annotatør» før claim.");
      return;
    }
    setLoading(true);
    setMsg("");
    try {
      const r = await fetch(`${API_BASE}/api/reviews/claim-next`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ annotator_id: aid }),
      });
      const raw = await r.json();
      if (!r.ok) throw new Error(typeof raw?.detail === "string" ? raw.detail : JSON.stringify(raw));
      await load();
      setIdx(0);
      setMsg("Claimet neste ledige element.");
    } catch (e) {
      setMsg(e instanceof Error ? e.message : "Feil ved claim");
    } finally {
      setLoading(false);
    }
  }, [annotatorId, load]);

  const submitRef = useRef(submit);
  submitRef.current = submit; // alltid siste submit for tastatur

  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      const el = e.target as HTMLElement | null;
      if (el?.closest("input, textarea, select, button, a")) return;
      if (e.key === "a" || e.key === "A") void submitRef.current(true);
      if (e.key === "1") setAnnotationLabel("alarm_sign");
      if (e.key === "2") setAnnotationLabel("unclear");
      if (e.key === "3") setAnnotationLabel("not_alarm_sign");
      if (e.key === "n" || e.key === "N")
        setIdx((i) => Math.min(Math.max(0, queue.length - 1), i + 1));
      if (e.key === "p" || e.key === "P") setIdx((i) => Math.max(0, i - 1));
      if (e.key === "Enter") void submitRef.current(false);
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [queue.length]);

  if (!current) {
    return (
      <>
        <h1>Review-kø</h1>
        <p>{queue.length === 0 && !msg ? "Ingen ventende — bra jobbet." : msg}</p>
        <button type="button" className="secondary" onClick={load}>
          Oppdater
        </button>
      </>
    );
  }

  const p = current.prediction;
  const im = current.image;

  return (
    <>
      <h1>Review-kø</h1>
      <div className="card" style={{ marginBottom: "1rem", padding: "0.75rem 1rem" }}>
        <label style={{ display: "block", marginBottom: 8 }}>
          Annotatør (navn / initialer — lagres med hver annotering)
          <input
            type="text"
            value={annotatorId}
            onChange={(e) => setAnnotatorId(e.target.value)}
            placeholder="f.eks. Kari"
            style={{ display: "block", marginTop: 4, maxWidth: 280, width: "100%" }}
          />
        </label>
        {queueStats && (
          <p className="muted" style={{ fontSize: 13, margin: "0.25rem 0" }}>
            Kø: {queueStats.pending_review} venter · {queueStats.free_or_expired_claim} ledig / utløpt claim ·{" "}
            {queueStats.claimed_active} aktivt claimt · {queueStats.completed_total} ferdig totalt
          </p>
        )}
        <p style={{ margin: 0, display: "flex", gap: 8, flexWrap: "wrap", alignItems: "center" }}>
          <button type="button" className="secondary" disabled={loading} onClick={() => void claimNext()}>
            Claim neste
          </button>
          <button type="button" className="secondary" disabled={loading} onClick={() => void load()}>
            Oppdater kø
          </button>
        </p>
      </div>
      <p className="muted">
        Datasett: <kbd>1</kbd> alarm-skilt · <kbd>2</kbd> uklart · <kbd>3</kbd> ikke alarm-skilt.{" "}
        <kbd>A</kbd> godkjenn modellforslag som status, <kbd>Enter</kbd> lagre merking, <kbd>P</kbd>/
        <kbd>N</kbd> forrige/neste.
      </p>
      <div className="grid2">
        <div className="card">
          <ReviewBboxEditor
            imageUrl={fileUrl(im.id, "original")}
            modelBbox={p.bbox_json}
            value={manualBbox}
            onChange={setManualBbox}
          />
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src={fileUrl(im.id, "evidence")}
            alt="evidens"
            style={{ maxWidth: "100%", marginTop: 8 }}
            onError={(e) => {
              (e.target as HTMLImageElement).style.display = "none";
            }}
          />
        </div>
        <div className="card">
          <p>
            <strong>Fil:</strong> {im.original_filename}
          </p>
          <p className="muted">
            Prediksjon #{p.id} · Bilde #{im.id}
            {p.claimed_by && (
              <>
                <br />
                Claimt av: <strong>{p.claimed_by}</strong>
                {p.claimed_at && <span> ({p.claimed_at})</span>}
              </>
            )}
          </p>
          <p>
            <strong>Modellforslag (ikke auto skilt_funnet):</strong> {p.predicted_status} ({p.confidence}%)
          </p>
          {p.bbox_json && (
            <p className="muted" style={{ fontSize: 12 }}>
              Bbox-forslag (0–1): {JSON.stringify(p.bbox_json)}
            </p>
          )}
          <p className="muted">{p.rationale}</p>
          <label>
            Treningslabel
            <select
              style={{ display: "block", marginTop: 4, width: "100%" }}
              value={annotationLabel}
              onChange={(e) => setAnnotationLabel(e.target.value as AnnotationLabel)}
            >
              {ANNOTATION_LABELS.map((s) => (
                <option key={s.value} value={s.value}>
                  {s.label}
                </option>
              ))}
            </select>
          </label>
          <label>
            YOLO datasett-split
            <select
              style={{ display: "block", marginTop: 4, width: "100%" }}
              value={yoloSplit}
              onChange={(e) => setYoloSplit(e.target.value as typeof yoloSplit)}
            >
              <option value="train">train</option>
              <option value="val">val</option>
              <option value="rejected">rejected / excluded</option>
              <option value="">(ikke i YOLO-eksport)</option>
            </select>
          </label>
          <p className="muted" style={{ marginTop: 6, fontSize: 12 }}>
            Lagres som status: {reviewStatusForAnnotation(annotationLabel)} · bbox: manuell hvis tegnet,
            ellers modellforslag.
          </p>
          <label style={{ display: "block", marginTop: 8 }}>
            Feiltype (ved overstyring)
            <select
              style={{ display: "block", marginTop: 4, width: "100%" }}
              value={errorType}
              onChange={(e) => setErrorType(e.target.value)}
            >
              <option value="">—</option>
              {ERROR_TYPES.map((s) => (
                <option key={s.value} value={s.value}>
                  {s.label}
                </option>
              ))}
            </select>
          </label>
          <textarea
            placeholder="Kommentar"
            value={comment}
            onChange={(e) => setComment(e.target.value)}
            style={{ width: "100%", marginTop: 8, minHeight: 70 }}
          />
          <p style={{ marginTop: 8, display: "flex", gap: 8, flexWrap: "wrap" }}>
            <button type="button" disabled={loading} onClick={() => submit(true)}>
              Godkjenn foreslått (A)
            </button>
            <button type="button" disabled={loading} onClick={() => submit(false)}>
              Lagre valgt status (Enter)
            </button>
            <span className="muted" style={{ alignSelf: "center" }}>
              {idx + 1} / {queue.length}
            </span>
          </p>
          {msg && <p>{msg}</p>}
        </div>
      </div>
    </>
  );
}
