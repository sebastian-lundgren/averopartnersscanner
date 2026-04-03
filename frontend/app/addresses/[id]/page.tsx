import Link from "next/link";
import { apiGet } from "@/lib/api";
import BestViewButton from "./BestViewButton";
import FinalStatusForm from "./FinalStatusForm";

type Addr = {
  id: number;
  customer_id: string | null;
  address_line: string | null;
  attempt_count: number;
  best_quality_score: number | null;
  selected_image_id: number | null;
  final_human_status: string | null;
  selection_metadata_json: Record<string, unknown> | null;
};

type Img = {
  id: number;
  original_filename: string;
  is_primary_for_address: boolean;
  is_temporary_candidate: boolean;
  quality_score: number | null;
  discard_reason: string | null;
};

export default async function AddressDetailPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  const aid = Number(id);
  let addr: Addr | null = null;
  let images: Img[] = [];
  try {
    addr = await apiGet<Addr>(`/api/addresses/${aid}`);
    images = await apiGet<Img[]>(`/api/addresses/${aid}/images`);
  } catch {
    addr = null;
  }
  if (!addr) return <p>Ikke funnet</p>;

  return (
    <>
      <p>
        <Link href="/addresses">← Adresser</Link>
      </p>
      <h1>
        Adresse #{addr.id} {addr.address_line && `— ${addr.address_line}`}
      </h1>
      <p className="muted">Kunde-ID: {addr.customer_id ?? "—"}</p>
      <div className="card">
        <p>
          Forsøk (best view-kjøringer): {addr.attempt_count} — beste kvalitetsscore:{" "}
          {addr.best_quality_score != null ? addr.best_quality_score.toFixed(3) : "—"}
        </p>
        <p>Valgt hovedbilde-ID: {addr.selected_image_id ?? "—"}</p>
        {addr.selection_metadata_json && (
          <details>
            <summary>Metadata utvelgelse</summary>
            <pre style={{ fontSize: 12, overflow: "auto" }}>
              {JSON.stringify(addr.selection_metadata_json, null, 2)}
            </pre>
          </details>
        )}
        <BestViewButton addressId={addr.id} />
      </div>
      <FinalStatusForm addressId={addr.id} initial={addr.final_human_status} />
      <h2>Bilder på adressen</h2>
      <ul>
        {images.map((im) => (
          <li key={im.id}>
            <Link href={`/library/${im.id}`}>#{im.id}</Link> {im.original_filename}
            {im.is_primary_for_address ? " (primær)" : ""}
            {im.is_temporary_candidate ? " (kandidat)" : ""}
            {im.discard_reason && <span className="muted"> — {im.discard_reason}</span>}
          </li>
        ))}
      </ul>
    </>
  );
}
