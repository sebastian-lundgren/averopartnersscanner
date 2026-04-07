"use client";

import Link from "next/link";
import { useCallback, useEffect, useState } from "react";
import { API_BASE, apiPost } from "@/lib/api";

const DEFAULT_POSTCODE = "0154";
const DEFAULT_MAX_LOCATIONS = 5;
const DEFAULT_MAX_ATTEMPTS = 4;
const DEFAULT_MAX_IMAGES = 4;

type AddressOutcome = {
  order: number;
  location_id: number;
  address: string;
  final_result: string | null;
  notes: string | null;
  images_saved: number;
};

type ResultSummary = {
  scan_run_id: number;
  run_status: string;
  total_locations: number;
  completed_locations: number;
  locations_with_detection: number;
  images_saved: number;
  image_ids: number[];
  predictions_pending_review: number;
  address_outcomes?: AddressOutcome[];
  image_debug?: Array<{
    image_id: number;
    bbox_count: number;
    used_stored_path: string | null;
    annotated: boolean;
  }>;
};

type LocationsPlan = {
  source: string;
  postcode: string;
  unique_address_count: number;
  planned_count: number;
  truncated_to_max_locations: boolean;
  warnings: string[];
  rows: Array<{
    order: number;
    address: string;
    postcode: string;
    latitude: number;
    longitude: number;
  }>;
};

type Job = {
  id: number;
  status: string;
  postcode: string;
  max_locations: number;
  max_attempts: number;
  max_images_per_address?: number;
  locations_json_path: string;
  error_message: string | null;
  created_at: string;
  started_at: string | null;
  finished_at: string | null;
  scan_run_id: number | null;
  result_summary: ResultSummary | null;
  locations_plan?: LocationsPlan | null;
};

function outcomeLabel(o: AddressOutcome): string {
  if (o.images_saved > 0) return "Lagrede bilder";
  const n = (o.notes || "").toLowerCase();
  if (n.includes("hopper til neste")) return "Hoppet over (SV ubrukelig)";
  if (o.final_result === "detection_found") return "Treff (runner)";
  if (o.final_result === "no_hit") return "Fullført uten lagrede bilder";
  return o.final_result || "—";
}

export default function ScannerPage() {
  const [postcode, setPostcode] = useState(DEFAULT_POSTCODE);
  const [maxLocations, setMaxLocations] = useState(String(DEFAULT_MAX_LOCATIONS));
  const [maxAttempts, setMaxAttempts] = useState(String(DEFAULT_MAX_ATTEMPTS));
  const [maxImages, setMaxImages] = useState(String(DEFAULT_MAX_IMAGES));
  const [useDynamicLocations, setUseDynamicLocations] = useState(true);

  const [job, setJob] = useState<Job | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const loadLatest = useCallback(async () => {
    const r = await fetch(`${API_BASE}/api/streetview-scan-jobs?limit=1`, { cache: "no-store" });
    if (!r.ok) return;
    const arr = (await r.json()) as Job[];
    if (Array.isArray(arr) && arr[0]) setJob(arr[0]);
  }, []);

  useEffect(() => {
    void loadLatest();
  }, [loadLatest]);

  const refresh = useCallback(async (id: number) => {
    const r = await fetch(`${API_BASE}/api/streetview-scan-jobs/${id}`, { cache: "no-store" });
    if (!r.ok) throw new Error(await r.text());
    const j = (await r.json()) as Job;
    setJob(j);
    if (j.status === "done" || j.status === "failed") void loadLatest();
  }, [loadLatest]);

  useEffect(() => {
    if (!job || job.status === "done" || job.status === "failed") return;
    const t = setInterval(() => void refresh(job.id).catch(() => {}), 2000);
    return () => clearInterval(t);
  }, [job, refresh]);

  async function start() {
    setErr(null);
    const ml = parseInt(maxLocations, 10);
    const ma = parseInt(maxAttempts, 10);
    const mi = parseInt(maxImages, 10);
    if (!postcode.trim()) {
      setErr("Postnummer kan ikke være tomt.");
      return;
    }
    if (Number.isNaN(ml) || Number.isNaN(ma) || Number.isNaN(mi)) {
      setErr("Tallfelt må være gyldige heltall.");
      return;
    }
    setBusy(true);
    try {
      const j = await apiPost<Job>("/api/streetview-scan-jobs/start", {
        postcode: postcode.trim(),
        max_locations: ml,
        max_attempts: ma,
        max_images_per_address: mi,
        use_dynamic_locations: useDynamicLocations,
      });
      setJob(j);
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }

  const s = job?.result_summary;
  const hasFindings = Boolean(s && (s.images_saved > 0 || s.locations_with_detection > 0));
  const reviewHref =
    s && s.image_ids.length > 0 ? `/review?image_ids=${encodeURIComponent(s.image_ids.join(","))}` : "/review";

  const jobImages = job?.max_images_per_address ?? DEFAULT_MAX_IMAGES;
  const maxFromPlan = job?.locations_plan?.unique_address_count ?? 0;
  const canUseMaxFromPlan = maxFromPlan > 0;

  return (
    <div className="page">
      <h1>Google Street View-scan</h1>
      <p className="muted">
        Starter en bakgrunnsjobb som kjører den eksisterende Playwright/YOLO-runneren mot API-et (samme flyt som{" "}
        <code>python -m runner</code>). Standard hentes <strong>gate + husnummer</strong> for postnummeret fra{" "}
        <strong>OpenStreetMap (Overpass)</strong>, sortert alfabetisk på gate og på husnummer. Dekning er ikke som
        Matrikkelen — mangler OSM-tags, får du ingen rader. Hyppig bruk: vær snill med Overpass-serveren.
      </p>

      <div className="card" style={{ padding: 16, maxWidth: 420, marginBottom: 16 }}>
        <p style={{ marginTop: 0, fontWeight: 600 }}>Innstillinger før start</p>
        <label style={{ display: "block", marginBottom: 10, fontSize: 14 }}>
          Postnummer
          <input
            type="text"
            value={postcode}
            onChange={(e) => setPostcode(e.target.value)}
            style={{ display: "block", width: "100%", maxWidth: 200, marginTop: 4 }}
          />
        </label>
        <label style={{ display: "block", marginBottom: 10, fontSize: 14 }}>
          Maks antall adresser
          <div style={{ display: "flex", alignItems: "center", gap: 8, marginTop: 4 }}>
            <input
              type="number"
              min={1}
              max={500}
              value={maxLocations}
              onChange={(e) => setMaxLocations(e.target.value)}
              style={{ display: "block", width: "100%", maxWidth: 200 }}
            />
            <button
              type="button"
              className="secondary"
              disabled={!canUseMaxFromPlan}
              onClick={() => setMaxLocations(String(maxFromPlan))}
            >
              Maks
            </button>
          </div>
        </label>
        <label style={{ display: "block", marginBottom: 10, fontSize: 14 }}>
          Maks antall forsøk per adresse
          <input
            type="number"
            min={1}
            max={20}
            value={maxAttempts}
            onChange={(e) => setMaxAttempts(e.target.value)}
            style={{ display: "block", width: "100%", maxWidth: 200, marginTop: 4 }}
          />
        </label>
        <label style={{ display: "block", marginBottom: 10, fontSize: 14 }}>
          Maks antall bilder per adresse
          <input
            type="number"
            min={1}
            max={20}
            value={maxImages}
            onChange={(e) => setMaxImages(e.target.value)}
            style={{ display: "block", width: "100%", maxWidth: 200, marginTop: 4 }}
          />
        </label>
        <label style={{ display: "flex", gap: 8, alignItems: "center", marginBottom: 12, fontSize: 14, cursor: "pointer" }}>
          <input
            type="checkbox"
            checked={useDynamicLocations}
            onChange={(e) => setUseDynamicLocations(e.target.checked)}
          />
          Hent adresser (gate+nr) fra OpenStreetMap Overpass for postnummer (anbefalt)
        </label>
        {!useDynamicLocations ? (
          <p className="muted" style={{ fontSize: 12, marginTop: 0 }}>
            Da brukes statisk JSON (<code>runner/data/example_locations.json</code> eller{" "}
            <code>GSV_SCAN_LOCATIONS_PATH</code>) — kun rader som matcher postnummeret du skrev.
          </p>
        ) : null}
        <button type="button" className="primary" disabled={busy} onClick={() => void start()}>
          Start Google Street View-scan
        </button>
      </div>

      {err && <p style={{ color: "var(--warn)" }}>{err}</p>}

      {job && (
        <div className="card" style={{ padding: 16, maxWidth: 920, marginTop: 16 }}>
          <h2 style={{ marginTop: 0, fontSize: "1.1rem" }}>Siste scan-jobb</h2>
          <p style={{ marginBottom: 8 }}>
            <strong>Jobb #{job.id}</strong> · status: <code>{job.status}</code>
            {job.scan_run_id != null ? (
              <>
                {" "}
                · ScanRun <code>#{job.scan_run_id}</code>
              </>
            ) : null}
          </p>
          <div
            style={{
              marginBottom: 12,
              padding: 10,
              background: "var(--surface)",
              borderRadius: 6,
              border: "1px solid var(--border)",
            }}
          >
            <p style={{ margin: "0 0 6px", fontWeight: 600, fontSize: 13 }}>Verdier brukt for denne jobben</p>
            <ul className="muted" style={{ fontSize: 13, margin: 0, paddingLeft: 18 }}>
              <li>
                Postnummer: <code>{job.postcode}</code>
              </li>
              <li>
                Maks adresser: <code>{job.max_locations}</code>
              </li>
              <li>
                Maks forsøk per adresse: <code>{job.max_attempts}</code>
              </li>
              <li>
                Maks bilder per adresse: <code>{jobImages}</code>
              </li>
              <li>
                Lokasjonskilde:{" "}
                {job.locations_json_path === "__dynamic__" ? (
                  <>OpenStreetMap Overpass — adresser sortert gate / husnummer (JSON ved start)</>
                ) : (
                  <>
                    Fil <code>{job.locations_json_path}</code>
                  </>
                )}
              </li>
            </ul>
          </div>

          {job.locations_plan && (
            <div
              style={{
                marginTop: 12,
                padding: 12,
                background: "var(--surface)",
                borderRadius: 6,
                border: "1px solid var(--border)",
              }}
            >
              <p style={{ margin: "0 0 8px", fontWeight: 600, fontSize: 14 }}>Adresser for denne jobben (Overpass / plan)</p>
              <p className="muted" style={{ fontSize: 13, margin: "0 0 8px" }}>
                Unike adresser i kilden for postnummeret: <strong>{job.locations_plan.unique_address_count}</strong>
                {" · "}
                Valgt til denne kjøringen: <strong>{job.locations_plan.planned_count}</strong> (maks {job.max_locations}
                {job.locations_plan.truncated_to_max_locations ? ", avkortet" : ""})
                {job.locations_plan.source === "overpass" ? " · kilde: OSM Overpass" : " · kilde: JSON-fil"}
              </p>
              {job.locations_plan.warnings.length > 0 ? (
                <ul className="muted" style={{ fontSize: 12, margin: "0 0 10px", paddingLeft: 18, color: "var(--warn)" }}>
                  {job.locations_plan.warnings.map((w, i) => (
                    <li key={i}>{w}</li>
                  ))}
                </ul>
              ) : null}
              <p style={{ margin: "0 0 6px", fontWeight: 600, fontSize: 12 }}>Rekkefølge (samme som runner)</p>
              <div style={{ overflowX: "auto", maxHeight: 280, overflowY: "auto" }}>
                <table style={{ width: "100%", fontSize: 12, borderCollapse: "collapse" }}>
                  <thead>
                    <tr style={{ textAlign: "left", borderBottom: "1px solid var(--border)" }}>
                      <th style={{ padding: "4px 6px" }}>#</th>
                      <th style={{ padding: "4px 6px" }}>Adresse</th>
                      <th style={{ padding: "4px 6px" }}>Koord</th>
                    </tr>
                  </thead>
                  <tbody>
                    {job.locations_plan.rows.map((r) => (
                      <tr key={r.order} style={{ borderBottom: "1px solid var(--border)" }}>
                        <td style={{ padding: "4px 6px", verticalAlign: "top" }}>{r.order}</td>
                        <td style={{ padding: "4px 6px", verticalAlign: "top" }}>{r.address}</td>
                        <td style={{ padding: "4px 6px", verticalAlign: "top", whiteSpace: "nowrap" }} className="muted">
                          {r.latitude.toFixed(5)}, {r.longitude.toFixed(5)}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {job.status === "done" && s && s.address_outcomes && s.address_outcomes.length > 0 && (
            <div style={{ marginTop: 14 }}>
              <p style={{ margin: "0 0 6px", fontWeight: 600, fontSize: 14 }}>Resultat per adresse (etter kjøring)</p>
              <div style={{ overflowX: "auto" }}>
                <table style={{ width: "100%", fontSize: 12, borderCollapse: "collapse" }}>
                  <thead>
                    <tr style={{ textAlign: "left", borderBottom: "1px solid var(--border)" }}>
                      <th style={{ padding: "4px 6px" }}>#</th>
                      <th style={{ padding: "4px 6px" }}>Adresse</th>
                      <th style={{ padding: "4px 6px" }}>Status</th>
                      <th style={{ padding: "4px 6px" }}>Bilder</th>
                      <th style={{ padding: "4px 6px" }}>Notat</th>
                    </tr>
                  </thead>
                  <tbody>
                    {s.address_outcomes.map((o) => (
                      <tr key={o.location_id} style={{ borderBottom: "1px solid var(--border)" }}>
                        <td style={{ padding: "4px 6px", verticalAlign: "top" }}>{o.order}</td>
                        <td style={{ padding: "4px 6px", verticalAlign: "top" }}>
                          {o.address} <span className="muted">loc #{o.location_id}</span>
                        </td>
                        <td style={{ padding: "4px 6px", verticalAlign: "top" }}>{outcomeLabel(o)}</td>
                        <td style={{ padding: "4px 6px", verticalAlign: "top" }}>{o.images_saved}</td>
                        <td style={{ padding: "4px 6px", verticalAlign: "top" }} className="muted">
                          {o.notes || "—"}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {job.status === "done" && s && (
            <div style={{ marginTop: 12, paddingTop: 12, borderTop: "1px solid var(--border)" }}>
              <p style={{ margin: "0 0 8px", fontWeight: 600 }}>
                {hasFindings
                  ? "Scan fant treff — bilder er lagret og kan gå til review."
                  : "Ingen deteksjon med lagrede bilder i denne kjøringen (sjekk ScanRun-tellinger under)."}
              </p>
              <ul className="muted" style={{ fontSize: 13, margin: "0 0 12px", paddingLeft: 18 }}>
                <li>
                  Lokasjoner fullført: {s.completed_locations} / {s.total_locations} (run-status: {s.run_status})
                </li>
                <li>Lokasjoner med deteksjon (telling fra run): {s.locations_with_detection}</li>
                <li>Lagrede bilder (detection hits med image_id): {s.images_saved}</li>
                <li>Prediksjoner som venter review (for disse bildene): {s.predictions_pending_review}</li>
              </ul>
              {(hasFindings || s.predictions_pending_review > 0) && (
                <div>
                  <p style={{ display: "flex", flexWrap: "wrap", gap: 8, alignItems: "center" }}>
                    <Link className="primary" href={reviewHref} style={{ padding: "6px 12px", borderRadius: 6 }}>
                      Gå til review-kø for disse bildene
                    </Link>
                    {s.image_ids.slice(0, 6).map((id) => (
                      <Link key={id} href={`/library/${id}`} className="secondary" style={{ padding: "6px 10px" }}>
                        Bibliotek #{id}
                      </Link>
                    ))}
                    {s.image_ids.length > 6 ? (
                      <span className="muted" style={{ fontSize: 12 }}>
                        +{s.image_ids.length - 6} flere (bruk review-lenken)
                      </span>
                    ) : null}
                  </p>
                  {s.image_debug && s.image_debug.length > 0 ? (
                    <div className="muted" style={{ fontSize: 12, marginTop: 8 }}>
                      {s.image_debug.map((d) => (
                        <div key={d.image_id}>
                          image #{d.image_id} · bbox_count={d.bbox_count} · annotated={d.annotated ? "yes" : "no"} · used_stored_path=
                          {d.used_stored_path || "(null)"}
                        </div>
                      ))}
                    </div>
                  ) : null}
                </div>
              )}
              {!hasFindings && s.predictions_pending_review === 0 ? (
                <p className="muted" style={{ fontSize: 13 }}>
                  <Link href="/review">Åpne hele review-køen</Link> eller{" "}
                  <Link href="/library">bildebibliotek</Link> om du vil fortsette manuelt.
                </p>
              ) : null}
            </div>
          )}

          {job.status === "done" && !s && (
            <p className="muted" style={{ fontSize: 13, marginTop: 12 }}>
              Ingen resultat-oppsummering (mangler koblet ScanRun-id i logg — sjekk at runner skriver «ScanRun &lt;id&gt;
              med …»).
            </p>
          )}

          {job.error_message && (
            <pre style={{ fontSize: 11, whiteSpace: "pre-wrap", overflow: "auto", marginTop: 12 }}>
              {job.error_message}
            </pre>
          )}
        </div>
      )}
    </div>
  );
}
