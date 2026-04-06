"use client";

import Link from "next/link";
import { useSearchParams } from "next/navigation";
import { useEffect, useMemo, useState } from "react";
import { API_BASE, fileUrl } from "@/lib/api";

type ImageRow = {
  id: number;
  address_id: number | null;
  original_filename: string;
  is_primary_for_address: boolean;
  is_temporary_candidate: boolean;
  display_name?: string;
  address_final_status?: string | null;
};

const HOME_FILTERS: { param: string; label: string }[] = [
  { param: "all", label: "Alle" },
  { param: "skilt_funnet", label: "Har alarm" },
  { param: "trenger_manuell", label: "Har ikke alarm" },
  { param: "uklart", label: "Uklart" },
];

export default function LibraryGallery() {
  const searchParams = useSearchParams();
  const raw = (searchParams.get("home_status") || "all").trim().toLowerCase();
  const homeStatus = HOME_FILTERS.some((f) => f.param === raw) ? raw : "all";

  const qs = useMemo(
    () =>
      homeStatus === "all"
        ? "limit=200"
        : `limit=200&home_status=${encodeURIComponent(homeStatus)}`,
    [homeStatus],
  );

  const [rows, setRows] = useState<ImageRow[]>([]);
  const [err, setErr] = useState("");
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setErr("");
    const url = `${API_BASE}/api/images/library?${qs}`;
    void (async () => {
      try {
        const r = await fetch(url, { cache: "no-store" });
        if (!r.ok) {
          const t = await r.text();
          if (!cancelled) setErr(t || r.statusText);
          return;
        }
        const data = (await r.json()) as ImageRow[];
        if (!cancelled) setRows(Array.isArray(data) ? data : []);
      } catch (e) {
        if (!cancelled) setErr(e instanceof Error ? e.message : "Feil");
      } finally {
        if (!cancelled) setLoading(false);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [qs]);

  return (
    <>
      <h1>Bildebibliotek</h1>
      <p className="muted" style={{ fontSize: 14, marginBottom: 12 }}>
        Filter etter registrert boligstatus (ikke modellforslag):{" "}
        {HOME_FILTERS.map((f, i) => (
          <span key={f.param}>
            {i > 0 ? " · " : null}
            {f.param === homeStatus ? (
              <strong>{f.label}</strong>
            ) : (
              <Link
                href={f.param === "all" ? "/library" : `/library?home_status=${encodeURIComponent(f.param)}`}
                prefetch={false}
              >
                {f.label}
              </Link>
            )}
          </span>
        ))}
      </p>
      {loading && <p className="muted">Laster …</p>}
      {err && <p className="muted">{err}</p>}
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fill, minmax(200px, 1fr))",
          gap: "0.75rem",
        }}
      >
        {rows.map((img) => {
          const title =
            img.display_name?.trim() || img.original_filename?.trim() || `Bilde-ID ${img.id}`;
          return (
            <div key={img.id} className="card" style={{ padding: 8 }}>
              <div style={{ position: "relative", borderRadius: 4, overflow: "hidden" }}>
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img
                  src={fileUrl(img.id, "original")}
                  alt=""
                  style={{
                    width: "100%",
                    height: 120,
                    objectFit: "cover",
                    display: "block",
                    verticalAlign: "top",
                  }}
                />
                <Link
                  href={`/library/${img.id}`}
                  prefetch={false}
                  aria-label={title}
                  style={{ position: "absolute", inset: 0, zIndex: 1 }}
                />
              </div>
              <div style={{ fontSize: 12, marginTop: 6, lineHeight: 1.35 }}>
                <span style={{ fontWeight: 600 }}>{title}</span>
              </div>
              {img.is_primary_for_address && <span style={{ fontSize: 11 }}>Primær for adresse</span>}
              {img.is_temporary_candidate && (
                <span style={{ fontSize: 11, color: "var(--warn)" }}> Kandidat</span>
              )}
            </div>
          );
        })}
      </div>
    </>
  );
}
