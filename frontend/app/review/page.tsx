"use client";

import { Suspense, useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import { useSearchParams } from "next/navigation";
import ReviewBboxEditor from "@/components/ReviewBboxEditor";
import type { BboxNorm } from "@/components/ReviewBboxEditor";
import { API_BASE, fileUrl } from "@/lib/api";
import { parsePredBboxes, parseYoloTrustMeta, yoloSuggestionsUncertain } from "@/lib/predBboxes";
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
  bbox_json: unknown;
  claimed_by?: string | null;
  claimed_at?: string | null;
};
type Img = { id: number; original_filename: string };
type QueueItem = { prediction: Pred; image: Img };

const ANNOTATOR_LS = "annotator_display_name";
const REVIEW_PAGE_SIZE = 100;
const REVIEW_TOTAL_CAP = 1000;

function ReviewPageInner() {
  const searchParams = useSearchParams();
  const imageIdsFilter = (searchParams.get("image_ids") || "").trim();

  const [queue, setQueue] = useState<QueueItem[]>([]);
  const [queuePage, setQueuePage] = useState(0);
  const [queueHasMore, setQueueHasMore] = useState(false);
  const [loadingMoreQueue, setLoadingMoreQueue] = useState(false);
  const [idx, setIdx] = useState(0);
  const [annotationLabel, setAnnotationLabel] = useState<AnnotationLabel>("unclear");
  const [comment, setComment] = useState("");
  const [errorType, setErrorType] = useState("");
  const [msg, setMsg] = useState("");
  const [loading, setLoading] = useState(false);
  const [workingBboxes, setWorkingBboxes] = useState<BboxNorm[]>([]);
  /** Når true: workingBboxes er (delvis) tilpasset av bruker — ikke arv YOLO «usikker»-UI for rene modellforslag. */
  const [manualAnnotationSession, setManualAnnotationSession] = useState(false);
  const [selectedBoxI, setSelectedBoxI] = useState(0);
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
      const imgQs =
        imageIdsFilter !== "" ? `&image_ids=${encodeURIComponent(imageIdsFilter)}` : "";
      const [r, st] = await Promise.all([
        fetch(`${API_BASE}/api/reviews/queue?skip=0&limit=${REVIEW_PAGE_SIZE}${qs}${imgQs}`),
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
      setQueue(data.slice(0, REVIEW_TOTAL_CAP));
      setQueuePage(0);
      setQueueHasMore(data.length === REVIEW_PAGE_SIZE && data.length < REVIEW_TOTAL_CAP);
      setIdx(0);
      const firstStatus = data[0]?.prediction?.predicted_status;
      if (firstStatus) setAnnotationLabel(defaultAnnotationFromPredicted(firstStatus));
    } catch (e) {
      setQueue([]);
      setIdx(0);
      setMsg(e instanceof Error ? e.message : "Kunne ikke hente kø");
    }
  }, [annotatorId, imageIdsFilter]);

  const loadMoreQueue = useCallback(async () => {
    if (loadingMoreQueue || !queueHasMore || queue.length >= REVIEW_TOTAL_CAP) return;
    setLoadingMoreQueue(true);
    setMsg("");
    try {
      const qs =
        annotatorId.trim() !== ""
          ? `&annotator_id=${encodeURIComponent(annotatorId.trim())}`
          : "";
      const imgQs =
        imageIdsFilter !== "" ? `&image_ids=${encodeURIComponent(imageIdsFilter)}` : "";
      const nextPage = queuePage + 1;
      const skip = nextPage * REVIEW_PAGE_SIZE;
      const remaining = REVIEW_TOTAL_CAP - queue.length;
      const limit = Math.min(REVIEW_PAGE_SIZE, remaining);
      const r = await fetch(`${API_BASE}/api/reviews/queue?skip=${skip}&limit=${limit}${qs}${imgQs}`);
      const raw = await r.json();
      if (!r.ok) throw new Error(typeof raw?.detail === "string" ? raw.detail : JSON.stringify(raw));
      if (!Array.isArray(raw)) throw new Error("Uventet svar fra API (ikke en liste).");
      const data = raw as QueueItem[];
      setQueue((prev) => [...prev, ...data].slice(0, REVIEW_TOTAL_CAP));
      setQueuePage(nextPage);
      setQueueHasMore(data.length === limit && skip + data.length < REVIEW_TOTAL_CAP);
    } catch (e) {
      setMsg(e instanceof Error ? e.message : "Kunne ikke hente flere i kø");
    } finally {
      setLoadingMoreQueue(false);
    }
  }, [annotatorId, imageIdsFilter, loadingMoreQueue, queueHasMore, queuePage, queue.length]);

  useEffect(() => {
    load();
  }, [load]);

  useEffect(() => {
    setIdx((i) => Math.min(i, Math.max(0, queue.length - 1)));
  }, [queue]);

  useEffect(() => {
    setSelectedBoxI((i) =>
      workingBboxes.length === 0 ? 0 : Math.min(i, Math.max(0, workingBboxes.length - 1))
    );
  }, [workingBboxes.length]);

  // Synkroniser bbox-liste før maling — unngår én frame med nytt bilde men gamle workingBboxes (feil «Boks 1/2»).
  useLayoutEffect(() => {
    const item = queue.length ? queue[idx] : undefined;
    if (!item) return;
    setAnnotationLabel(defaultAnnotationFromPredicted(item.prediction.predicted_status));
    const parsed = parsePredBboxes(item.prediction.bbox_json);
    setWorkingBboxes(parsed);
    setSelectedBoxI(0);
    setManualAnnotationSession(false);
  }, [queue, idx]);

  const modelBboxesParsed = useMemo(
    () => parsePredBboxes(queue.length ? queue[idx]?.prediction.bbox_json : undefined),
    [queue, idx],
  );

  const yoloTrustMeta = useMemo(
    () => parseYoloTrustMeta(queue.length ? queue[idx]?.prediction.bbox_json : undefined),
    [queue, idx],
  );

  const yoloUncertain = useMemo(
    () => yoloSuggestionsUncertain(queue.length ? queue[idx]?.prediction.bbox_json : undefined),
    [queue, idx],
  );

  const showYoloUncertainUi = yoloUncertain && !manualAnnotationSession;

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
        body.annotation_bboxes_json = workingBboxes;
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
    [queue, idx, annotationLabel, workingBboxes, yoloSplit, comment, errorType, annotatorId]
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

  const onBboxEditorChange = useCallback((b: BboxNorm | null) => {
    setManualAnnotationSession(true);
    if (b === null) {
      setWorkingBboxes((prev) => prev.filter((_, i) => i !== selectedBoxI));
      return;
    }
    setWorkingBboxes((prev) => {
      if (prev.length === 0) return [b];
      const n = [...prev];
      if (selectedBoxI >= 0 && selectedBoxI < n.length) {
        n[selectedBoxI] = b;
        return n;
      }
      return [...n, b];
    });
  }, [selectedBoxI]);

  const current = queue.length ? queue[idx] : undefined;

  if (!current) {
    return (
      <>
        <h1>Review-kø</h1>
        {imageIdsFilter ? (
          <p className="muted" style={{ fontSize: 13 }}>
            Filtrert kø: kun bilder fra valgt Street View-scan (image_ids).
          </p>
        ) : null}
        <p>{queue.length === 0 && !msg ? "Ingen ventende — bra jobbet." : msg}</p>
        <button type="button" className="secondary" onClick={load}>
          Oppdater
        </button>
      </>
    );
  }

  const p = current.prediction;
  const im = current.image;
  const originalPreviewUrl = fileUrl(im.id, "original");
  const evidencePreviewUrl = fileUrl(im.id, "evidence");
  const selectedBbox = workingBboxes[selectedBoxI] ?? null;
  const peerBboxes = workingBboxes.filter((_, i) => i !== selectedBoxI);

  return (
    <>
      <h1>Review-kø</h1>
      {imageIdsFilter ? (
        <p className="muted" style={{ fontSize: 13 }}>
          Filtrert kø: kun bilder fra valgt Street View-scan (image_ids).
        </p>
      ) : null}
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
          {showYoloUncertainUi && modelBboxesParsed.length > 0 && (
            <p
              className="muted"
              style={{
                fontSize: 13,
                marginBottom: 10,
                padding: "8px 10px",
                background: "var(--surface)",
                border: "1px solid var(--warn)",
                borderRadius: 6,
              }}
            >
              <strong>Usikre YOLO-forslag:</strong> modellen har ikke høy nok kvalitet på rangert primær
              (conf × heuristikk under terskel). Alle {modelBboxesParsed.length} bokser er kun forslag — ingen
              pålitelig auto-primær.
              {yoloTrustMeta.yolo_primary_gate_reason ? (
                <>
                  {" "}
                  <span style={{ fontSize: 12 }}>({yoloTrustMeta.yolo_primary_gate_reason})</span>
                </>
              ) : null}
            </p>
          )}
          <div
            style={{
              display: "flex",
              flexDirection: "row",
              alignItems: "stretch",
              gap: 8,
            }}
          >
            <button
              type="button"
              className="secondary"
              aria-label="Forrige bilde i køen"
              title="Forrige bilde"
              disabled={idx <= 0}
              onClick={() => setIdx((i) => Math.max(0, i - 1))}
              style={{
                alignSelf: "center",
                flexShrink: 0,
                minWidth: 44,
                minHeight: 120,
                padding: "4px 8px",
                fontSize: 28,
                lineHeight: 1,
              }}
            >
              ‹
            </button>
            <div style={{ flex: 1, minWidth: 0 }}>
          <div style={{ display: "flex", gap: 6, flexWrap: "wrap", marginBottom: 8, alignItems: "center" }}>
            {workingBboxes.map((_, i) => (
              <button
                key={i}
                type="button"
                className={i === selectedBoxI ? undefined : "secondary"}
                style={i === selectedBoxI ? { fontWeight: 600 } : undefined}
                onClick={() => setSelectedBoxI(i)}
              >
                Boks {i + 1}
                {showYoloUncertainUi ? " (usikker)" : ""}
              </button>
            ))}
            <button
              type="button"
              className="secondary"
              onClick={() => {
                setManualAnnotationSession(true);
                const nb: BboxNorm = { x: 0.38, y: 0.38, w: 0.14, h: 0.11 };
                let lastI = 0;
                setWorkingBboxes((prev) => {
                  const next = [...prev, nb];
                  lastI = next.length - 1;
                  return next;
                });
                setSelectedBoxI(lastI);
              }}
            >
              Legg til boks
            </button>
            <button
              type="button"
              className="secondary"
              disabled={workingBboxes.length === 0}
              onClick={() => {
                setManualAnnotationSession(true);
                setWorkingBboxes((prev) => prev.filter((_, i) => i !== selectedBoxI));
              }}
            >
              Slett valgt
            </button>
          </div>
          <ReviewBboxEditor
            key={p.id}
            imageUrl={originalPreviewUrl}
            modelBbox={selectedBbox}
            peerBboxes={peerBboxes}
            value={null}
            onChange={onBboxEditorChange}
            uncertainSuggestions={showYoloUncertainUi}
            onResetAllFromModel={() => {
              setManualAnnotationSession(false);
              setWorkingBboxes(parsePredBboxes(p.bbox_json));
              setSelectedBoxI(0);
            }}
          />
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            key={`evidence-thumb-${im.id}`}
            src={evidencePreviewUrl}
            alt="Evidensutsnitt"
            style={{ maxWidth: "100%", marginTop: 8 }}
            onError={(e) => {
              const el = e.currentTarget;
              if (!el.dataset.fallbackToOriginal) {
                el.dataset.fallbackToOriginal = "1";
                el.src = originalPreviewUrl;
                el.alt = "Original (evidens mangler eller ikke klar ennå)";
                return;
              }
              el.style.display = "none";
            }}
          />
            </div>
            <button
              type="button"
              className="secondary"
              aria-label="Neste bilde i køen"
              title="Neste bilde"
              disabled={idx >= queue.length - 1}
              onClick={() => setIdx((i) => Math.min(queue.length - 1, i + 1))}
              style={{
                alignSelf: "center",
                flexShrink: 0,
                minWidth: 44,
                minHeight: 120,
                padding: "4px 8px",
                fontSize: 28,
                lineHeight: 1,
              }}
            >
              ›
            </button>
          </div>
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
            {showYoloUncertainUi && modelBboxesParsed.length > 0 ? (
              <span className="muted" style={{ fontWeight: 600 }}>
                {" "}
                — rangert primær er markert som <em>usikker</em> (ikke pålitelig auto-valg).
              </span>
            ) : null}
          </p>
          {modelBboxesParsed.length > 0 && (
            <p className="muted" style={{ fontSize: 12 }}>
              Bbox-forslag ({modelBboxesParsed.length}
              {showYoloUncertainUi ? ", alle usikre" : ""}): {JSON.stringify(modelBboxesParsed)}
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
            Lagres som status: {reviewStatusForAnnotation(annotationLabel)} · bokser: {workingBboxes.length}{" "}
            (sendes som annotation_bboxes_json).
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
            {queueHasMore && (
              <button type="button" disabled={loadingMoreQueue} onClick={() => void loadMoreQueue()}>
                {loadingMoreQueue ? "Laster ..." : "Last flere"}
              </button>
            )}
          </p>
          {msg && <p>{msg}</p>}
        </div>
      </div>
    </>
  );
}

export default function ReviewPage() {
  return (
    <Suspense fallback={<div className="page"><p>Laster…</p></div>}>
      <ReviewPageInner />
    </Suspense>
  );
}
