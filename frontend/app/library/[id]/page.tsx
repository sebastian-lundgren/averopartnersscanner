import Link from "next/link";
import { apiGet, fileUrl } from "@/lib/api";
import LibraryDetailImage from "./LibraryDetailImage";
import TagLibraryForm from "./TagLibraryForm";

type ImageRow = {
  id: number;
  address_id: number | null;
  original_filename: string;
  evidence_crop_path: string | null;
};

export default async function LibraryDetailPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  const imageId = Number(id);
  let img: ImageRow | null = null;
  let preds: unknown[] = [];
  try {
    img = await apiGet<ImageRow>(`/api/images/${imageId}`);
    preds = await apiGet<unknown[]>(`/api/images/${imageId}/predictions`);
  } catch {
    img = null;
  }
  if (!img) return <p>Bilde ikke funnet.</p>;

  return (
    <>
      <p>
        <Link href="/library">← Tilbake</Link>
      </p>
      <h1>Bilde #{img.id}</h1>
      <div className="grid2">
        <div className="card">
          <LibraryDetailImage src={fileUrl(img.id, "original")} />
          {img.evidence_crop_path && (
            <div style={{ marginTop: 8 }}>
              <p className="muted">Evidensutsnitt</p>
              <LibraryDetailImage src={fileUrl(img.id, "evidence")} />
            </div>
          )}
        </div>
        <div className="card">
          <h3>Prediksjoner</h3>
          <pre style={{ overflow: "auto", fontSize: 12, minWidth: 0 }}>{JSON.stringify(preds, null, 2)}</pre>
          <TagLibraryForm imageId={img.id} />
        </div>
      </div>
    </>
  );
}
