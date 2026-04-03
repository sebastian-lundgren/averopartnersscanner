"use client";

import { useState } from "react";
import { API_BASE } from "@/lib/api";

export default function UploadPage() {
  const [addressId, setAddressId] = useState("");
  const [customerId, setCustomerId] = useState("");
  const [addressLine, setAddressLine] = useState("");
  const [candidate, setCandidate] = useState(false);
  const [msg, setMsg] = useState("");
  const [loading, setLoading] = useState(false);

  async function onSubmit(e: React.FormEvent<HTMLFormElement>) {
    e.preventDefault();
    setMsg("");
    const form = e.currentTarget;
    const input = form.elements.namedItem("files") as HTMLInputElement;
    if (!input.files?.length) {
      setMsg("Velg minst én fil.");
      return;
    }
    const fd = new FormData();
    for (const f of Array.from(input.files)) fd.append("files", f);
    if (addressId) fd.append("address_id", addressId);
    if (customerId) fd.append("customer_id", customerId);
    if (addressLine) fd.append("address_line", addressLine);
    fd.append("is_temporary_candidate", candidate ? "true" : "false");
    setLoading(true);
    try {
      const r = await fetch(`${API_BASE}/api/images/upload`, { method: "POST", body: fd });
      const j = await r.json();
      if (!r.ok) throw new Error(j.detail || JSON.stringify(j));
      setMsg(`OK: ${j.items?.length ?? 0} bilde(r) — adresse-id ${j.address_id ?? "—"}`);
      input.value = "";
    } catch (err) {
      setMsg(err instanceof Error ? err.message : "Feil");
    } finally {
      setLoading(false);
    }
  }

  return (
    <>
      <h1>Opplasting</h1>
      <p className="muted">
        Batch: velg flere filer. Mappe: de fleste nettlesere støtter mappevalg med{" "}
        <code>webkitdirectory</code> — bruk &quot;Velg filer&quot; og velg en mappe hvis nettleseren
        tillater det.
      </p>
      <form className="card" onSubmit={onSubmit}>
        <label>
          Filer (flere tillatt)
          <br />
          <input type="file" name="files" multiple accept="image/*" />
        </label>
        <p className="muted" style={{ marginTop: "0.5rem" }}>
          <label>
            <input
              type="file"
              name="folder"
              // @ts-expect-error webkitdirectory
              webkitdirectory=""
              directory=""
              multiple
              onChange={(ev) => {
                const inp = ev.target;
                const files = inp.files;
                if (!files?.length) return;
                const main = document.querySelector<HTMLInputElement>('input[name="files"]');
                if (main) {
                  const dt = new DataTransfer();
                  for (let i = 0; i < files.length; i++) dt.items.add(files[i]);
                  main.files = dt.files;
                }
                setMsg(`${files.length} filer fra mappe lagt i opplastingsfeltet.`);
              }}
            />{" "}
            Velg mappe (overfører til hovedfeltet)
          </label>
        </p>
        <hr style={{ borderColor: "var(--border)", margin: "1rem 0" }} />
        <label>
          Eksisterende adresse-ID (valgfritt)
          <input
            style={{ display: "block", width: "100%", marginTop: 4 }}
            value={addressId}
            onChange={(e) => setAddressId(e.target.value)}
            placeholder="f.eks. 1"
          />
        </label>
        <label style={{ display: "block", marginTop: 8 }}>
          Ny kunde-ID (valgfritt)
          <input
            style={{ display: "block", width: "100%", marginTop: 4 }}
            value={customerId}
            onChange={(e) => setCustomerId(e.target.value)}
          />
        </label>
        <label style={{ display: "block", marginTop: 8 }}>
          Adresselinje (valgfritt, oppretter ny adresse sammen med kunde)
          <input
            style={{ display: "block", width: "100%", marginTop: 4 }}
            value={addressLine}
            onChange={(e) => setAddressLine(e.target.value)}
          />
        </label>
        <label style={{ display: "block", marginTop: 12 }}>
          <input type="checkbox" checked={candidate} onChange={(e) => setCandidate(e.target.checked)} />{" "}
          Midlertidig kandidat for &quot;best view&quot;-utvelgelse
        </label>
        <p style={{ marginTop: 12 }}>
          <button type="submit" disabled={loading}>
            {loading ? "Laster…" : "Last opp"}
          </button>
        </p>
        {msg && <p>{msg}</p>}
      </form>
    </>
  );
}
