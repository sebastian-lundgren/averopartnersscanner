"use client";

import { API_BASE } from "@/lib/api";

export default function ExportPage() {
  const csv = `${API_BASE}/api/export/csv`;
  const xlsx = `${API_BASE}/api/export/xlsx`;
  return (
    <>
      <h1>Eksport</h1>
      <p className="muted">
        Felter: adresse/kunde, filnavn, foreslått og endelig status, confidence, overstyring,
        kommentar, feiltype, dato, modellversjon, evidenssti.
      </p>
      <div className="card" style={{ display: "flex", gap: 12, flexWrap: "wrap" }}>
        <a href={csv} download>
          <button type="button">Last ned CSV</button>
        </a>
        <a href={xlsx} download>
          <button type="button">Last ned Excel (.xlsx)</button>
        </a>
      </div>
    </>
  );
}
