import Link from "next/link";
import { apiGet, fileUrl } from "@/lib/api";

type ImageRow = {
  id: number;
  address_id: number | null;
  original_filename: string;
  is_primary_for_address: boolean;
  is_temporary_candidate: boolean;
};

export default async function LibraryPage() {
  let rows: ImageRow[] = [];
  let err = "";
  try {
    rows = await apiGet<ImageRow[]>("/api/images/library?limit=200");
  } catch (e) {
    err = e instanceof Error ? e.message : "Feil";
  }

  return (
    <>
      <h1>Bildebibliotek</h1>
      {err && <p className="muted">{err}</p>}
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fill, minmax(200px, 1fr))",
          gap: "0.75rem",
        }}
      >
        {rows.map((img) => (
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
                aria-label={`Åpne bilde ${img.id}`}
                style={{ position: "absolute", inset: 0, zIndex: 1 }}
              />
            </div>
            <div className="muted" style={{ fontSize: 12, marginTop: 6 }}>
              #{img.id} {img.original_filename}
            </div>
            {img.is_primary_for_address && <span style={{ fontSize: 11 }}>Primær for adresse</span>}
            {img.is_temporary_candidate && (
              <span style={{ fontSize: 11, color: "var(--warn)" }}> Kandidat</span>
            )}
          </div>
        ))}
      </div>
    </>
  );
}
