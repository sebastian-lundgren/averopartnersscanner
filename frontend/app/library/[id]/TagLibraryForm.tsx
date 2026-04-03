"use client";

import { useState } from "react";
import { API_BASE } from "@/lib/api";
import { LIBRARY_CATEGORIES } from "@/lib/constants";

export default function TagLibraryForm({ imageId }: { imageId: number }) {
  const [category, setCategory] = useState(LIBRARY_CATEGORIES[0].value);
  const [notes, setNotes] = useState("");
  const [msg, setMsg] = useState("");

  async function save() {
    setMsg("");
    const r = await fetch(`${API_BASE}/api/training/library/${imageId}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ category, notes: notes || null, tags: {} }),
    });
    const t = await r.text();
    if (!r.ok) {
      setMsg(t);
      return;
    }
    setMsg("Lagret i eksempelbibliotek");
  }

  return (
    <div style={{ marginTop: 16 }}>
      <h4>Eksempelbibliotek</h4>
      <select value={category} onChange={(e) => setCategory(e.target.value)}>
        {LIBRARY_CATEGORIES.map((c) => (
          <option key={c.value} value={c.value}>
            {c.label}
          </option>
        ))}
      </select>
      <textarea
        placeholder="Notat"
        value={notes}
        onChange={(e) => setNotes(e.target.value)}
        style={{ display: "block", width: "100%", marginTop: 8, minHeight: 60 }}
      />
      <button type="button" onClick={save} style={{ marginTop: 8 }}>
        Lagre kategori
      </button>
      {msg && <p className="muted">{msg}</p>}
    </div>
  );
}
