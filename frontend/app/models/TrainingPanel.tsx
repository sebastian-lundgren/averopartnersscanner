"use client";

import { useCallback, useEffect, useState } from "react";
import { API_BASE } from "@/lib/api";

type TrainJob = {
  id: number;
  status: string;
  trigger: string;
  created_at: string;
  started_at: string | null;
  finished_at: string | null;
  error_message: string | null;
  config_json: Record<string, unknown> | null;
  export_counts_json: Record<string, number> | null;
  metrics_json: Record<string, number> | null;
  new_annotations_snapshot: number | null;
  candidate_model_version_id: number | null;
  activated_new_model: boolean;
};

type AutoStatus = {
  new_annotations_since_checkpoint: number;
  trigger_threshold: number;
  auto_enabled: boolean;
  train_job_busy: boolean;
};

type Mv = {
  id: number;
  version_tag: string;
  description: string | null;
  is_active: boolean;
  weights_path: string | null;
  metrics_json: Record<string, number> | null;
  train_image_count: number | null;
};

export default function TrainingPanel({ versions }: { versions: Mv[] }) {
  const [jobs, setJobs] = useState<TrainJob[]>([]);
  const [auto, setAuto] = useState<AutoStatus | null>(null);
  const [msg, setMsg] = useState("");
  const [loading, setLoading] = useState(false);

  const refresh = useCallback(async () => {
    try {
      const [j, a] = await Promise.all([
        fetch(`${API_BASE}/api/train-jobs?limit=30`).then((r) => r.json()),
        fetch(`${API_BASE}/api/train-jobs/auto-trigger-status`).then((r) => r.json()),
      ]);
      setJobs(Array.isArray(j) ? j : []);
      setAuto(typeof a === "object" && a ? (a as AutoStatus) : null);
    } catch {
      setJobs([]);
      setAuto(null);
    }
  }, []);

  useEffect(() => {
    void refresh();
    const t = setInterval(() => void refresh(), 8000);
    return () => clearInterval(t);
  }, [refresh]);

  const startTrain = async () => {
    setLoading(true);
    setMsg("");
    try {
      const r = await fetch(`${API_BASE}/api/train-jobs/start`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({}),
      });
      const raw = await r.json();
      if (!r.ok) throw new Error(typeof raw?.detail === "string" ? raw.detail : JSON.stringify(raw));
      setMsg(`Jobb #${(raw as TrainJob).id} startet`);
      await refresh();
    } catch (e) {
      setMsg(e instanceof Error ? e.message : "Feil");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="card" style={{ marginBottom: "1rem" }}>
      <h2 style={{ marginTop: 0 }}>YOLO-trening</h2>
      {auto && (
        <p className="muted" style={{ fontSize: 14 }}>
          Auto-trening: {auto.auto_enabled ? "på" : "av"} · nye annoteringer siden siste vellykkede jobb:{" "}
          <strong>{auto.new_annotations_since_checkpoint}</strong> / terskel {auto.trigger_threshold}
          {auto.train_job_busy ? " · jobb pågår" : ""}
        </p>
      )}
      <p>
        <button type="button" disabled={loading} onClick={() => void startTrain()}>
          Train new YOLO model
        </button>{" "}
        <button type="button" className="secondary" onClick={() => void refresh()}>
          Oppdater status
        </button>
      </p>
      {msg && <p>{msg}</p>}
      <h3 style={{ fontSize: "1rem" }}>Treningsjobber</h3>
      <ul style={{ fontSize: 13, paddingLeft: "1.2rem" }}>
        {jobs.map((job) => (
          <li key={job.id} style={{ marginBottom: 6 }}>
            <strong>#{job.id}</strong> {job.status} ({job.trigger}) — opprettet {job.created_at}
            {job.metrics_json && (
              <span className="muted">
                {" "}
                · P {job.metrics_json.precision?.toFixed?.(3) ?? "?"} R{" "}
                {job.metrics_json.recall?.toFixed?.(3) ?? "?"} mAP50{" "}
                {job.metrics_json["mAP50"]?.toFixed?.(3) ?? "?"} mAP50-95{" "}
                {job.metrics_json["mAP50-95"]?.toFixed?.(3) ?? "?"}
              </span>
            )}
            {job.activated_new_model && <span className="muted"> · aktivert som inferensmodell</span>}
            {job.error_message && (
              <div style={{ color: "var(--err, #c00)", whiteSpace: "pre-wrap" }}>{job.error_message}</div>
            )}
          </li>
        ))}
        {jobs.length === 0 && <li className="muted">Ingen jobber ennå</li>}
      </ul>
      <h3 style={{ fontSize: "1rem" }}>Modellversjoner (YOLO-vekter)</h3>
      <ul style={{ fontSize: 13, paddingLeft: "1.2rem" }}>
        {versions
          .filter((v) => v.weights_path)
          .map((v) => (
            <li key={v.id}>
              <code>{v.version_tag}</code>
              {v.is_active ? " (DB aktiv — vanligvis DINO for prediksjoner)" : ""} · bilder:{" "}
              {v.train_image_count ?? "—"} · metrics: {v.metrics_json ? JSON.stringify(v.metrics_json) : "—"}
              <div className="muted" style={{ wordBreak: "break-all" }}>
                {v.weights_path}
              </div>
            </li>
          ))}
        {versions.filter((v) => v.weights_path).length === 0 && (
          <li className="muted">Ingen YOLO-vekter registrert i DB ennå (kjør trening)</li>
        )}
      </ul>
    </div>
  );
}
