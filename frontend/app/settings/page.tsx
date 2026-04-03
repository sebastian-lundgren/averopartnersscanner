"use client";

import { useCallback, useEffect, useState } from "react";
import { API_BASE, apiGet, apiPut } from "@/lib/api";

type Thr = {
  threshold_strong_sign: number;
  threshold_unclear_high: number;
  threshold_unclear_low: number;
  max_best_view_attempts: number;
  quality_threshold: number;
};

export default function SettingsPage() {
  const [t, setT] = useState<Thr | null>(null);
  const [msg, setMsg] = useState("");

  const load = useCallback(async () => {
    try {
      const data = await apiGet<Thr>("/api/settings/thresholds");
      setT(data);
    } catch (e) {
      setMsg(e instanceof Error ? e.message : "Feil");
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  async function save() {
    if (!t) return;
    setMsg("");
    try {
      await apiPut("/api/settings/thresholds", t);
      setMsg("Lagret");
    } catch (e) {
      setMsg(e instanceof Error ? e.message : "Feil");
    }
  }

  if (!t) return <p>Laster… {msg}</p>;

  return (
    <>
      <h1>Innstillinger</h1>
      <p className="muted">
        Terskler styrer når prediksjon flagges for review. Systemet bruker aldri kategorien «ingen
        alarm».
      </p>
      <div className="card">
        <label>
          Sterk skilt-mistanke (confidence ≥): {t.threshold_strong_sign}
          <input
            type="range"
            min={50}
            max={100}
            value={t.threshold_strong_sign}
            onChange={(e) => setT({ ...t, threshold_strong_sign: Number(e.target.value) })}
            style={{ width: "100%" }}
          />
        </label>
        <label>
          Uklart høy grense: {t.threshold_unclear_high}
          <input
            type="range"
            min={40}
            max={99}
            value={t.threshold_unclear_high}
            onChange={(e) => setT({ ...t, threshold_unclear_high: Number(e.target.value) })}
            style={{ width: "100%" }}
          />
        </label>
        <label>
          Uklart lav grense: {t.threshold_unclear_low}
          <input
            type="range"
            min={0}
            max={60}
            value={t.threshold_unclear_low}
            onChange={(e) => setT({ ...t, threshold_unclear_low: Number(e.target.value) })}
            style={{ width: "100%" }}
          />
        </label>
        <label>
          Maks forsøk best view per adresse: {t.max_best_view_attempts}
          <input
            type="range"
            min={3}
            max={8}
            value={t.max_best_view_attempts}
            onChange={(e) => setT({ ...t, max_best_view_attempts: Number(e.target.value) })}
            style={{ width: "100%" }}
          />
        </label>
        <label>
          Kvalitetsterskel (0–1) for best view: {t.quality_threshold.toFixed(2)}
          <input
            type="range"
            min={0.2}
            max={0.8}
            step={0.01}
            value={t.quality_threshold}
            onChange={(e) => setT({ ...t, quality_threshold: Number(e.target.value) })}
            style={{ width: "100%" }}
          />
        </label>
        <p>
          <button type="button" onClick={save}>
            Lagre
          </button>
        </p>
        {msg && <p>{msg}</p>}
        <p className="muted" style={{ marginTop: 16 }}>
          Backend: <code>{API_BASE}</code>
        </p>
      </div>
      <div className="card" style={{ marginTop: "1rem" }}>
        <h2>YOLO datasett / trening</h2>
        <p className="muted">
          Eksporter annoterte rader (train/val/rejected) til disk, eller start Ultralytics-trening i
          bakgrunnen.
        </p>
        <YoloActions />
      </div>
    </>
  );
}

function YoloActions() {
  const [m, setM] = useState("");
  async function exp() {
    setM("");
    try {
      const r = await fetch(`${API_BASE}/api/yolo/dataset/export-disk`, { method: "POST" });
      const j = await r.json();
      if (!r.ok) throw new Error(JSON.stringify(j));
      setM(`Eksportert: ${j.export_dir} — ${JSON.stringify(j.counts)}`);
    } catch (e) {
      setM(e instanceof Error ? e.message : "Feil");
    }
  }
  async function train() {
    setM("");
    try {
      const r = await fetch(`${API_BASE}/api/yolo/train`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ epochs: 30, batch: 4, name: "from_app" }),
      });
      const j = await r.json();
      if (!r.ok) throw new Error(JSON.stringify(j));
      setM(j.message || "Startet");
    } catch (e) {
      setM(e instanceof Error ? e.message : "Feil");
    }
  }
  return (
    <p style={{ display: "flex", gap: 8, flexWrap: "wrap", alignItems: "center" }}>
      <button type="button" className="secondary" onClick={() => void exp()}>
        Eksporter datasett til disk
      </button>
      <button type="button" onClick={() => void train()}>
        Start YOLO-trening (bakgrunn)
      </button>
      {m && <span className="muted">{m}</span>}
    </p>
  );
}
