"use client";

import { useState } from "react";
import { API_BASE } from "@/lib/api";
import { STATUSES } from "@/lib/constants";

export default function FinalStatusForm({
  addressId,
  initial,
}: {
  addressId: number;
  initial: string | null;
}) {
  const [v, setV] = useState(initial || "uklart");
  const [msg, setMsg] = useState("");
  async function save() {
    const r = await fetch(`${API_BASE}/api/addresses/${addressId}/final-status`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ final_human_status: v }),
    });
    setMsg(r.ok ? "Lagret" : await r.text());
  }
  return (
    <div className="card">
      <h3>Endelig menneskelig status for adresse</h3>
      <select value={v} onChange={(e) => setV(e.target.value)}>
        {STATUSES.map((s) => (
          <option key={s.value} value={s.value}>
            {s.label}
          </option>
        ))}
      </select>
      <button type="button" style={{ marginLeft: 8 }} onClick={save}>
        Lagre
      </button>
      {msg && <span className="muted"> {msg}</span>}
    </div>
  );
}
