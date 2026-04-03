"use client";

import { useState } from "react";
import { API_BASE } from "@/lib/api";

export default function BestViewButton({ addressId }: { addressId: number }) {
  const [msg, setMsg] = useState("");
  async function run() {
    setMsg("");
    const r = await fetch(`${API_BASE}/api/addresses/${addressId}/best-view`, { method: "POST" });
    const j = await r.json();
    setMsg(JSON.stringify(j, null, 2));
    if (r.ok) window.location.reload();
  }
  return (
    <p>
      <button type="button" className="secondary" onClick={run}>
        Kjør best view-utvelgelse (kandidatbilder)
      </button>
      {msg && (
        <pre style={{ marginTop: 8, fontSize: 12, whiteSpace: "pre-wrap" }}>{msg}</pre>
      )}
    </p>
  );
}
