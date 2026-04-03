import Link from "next/link";
import { apiGet } from "@/lib/api";

type Addr = {
  id: number;
  customer_id: string | null;
  address_line: string | null;
  image_count: number;
  attempt_count: number;
  best_quality_score: number | null;
  final_human_status: string | null;
};

export default async function AddressesPage() {
  let rows: Addr[] = [];
  let err = "";
  try {
    rows = await apiGet<Addr[]>("/api/addresses");
  } catch (e) {
    err = e instanceof Error ? e.message : "Feil";
  }

  return (
    <>
      <h1>Adresser / kunder</h1>
      {err && <p className="muted">{err}</p>}
      <div className="card">
        <table style={{ width: "100%", borderCollapse: "collapse" }}>
          <thead>
            <tr style={{ textAlign: "left", borderBottom: "1px solid var(--border)" }}>
              <th>ID</th>
              <th>Kunde</th>
              <th>Adresse</th>
              <th>Bilder</th>
              <th>Forsøk (best view)</th>
              <th>Beste kvalitet</th>
              <th>Godkjent status</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((a) => (
              <tr key={a.id} style={{ borderBottom: "1px solid var(--border)" }}>
                <td>
                  <Link href={`/addresses/${a.id}`}>{a.id}</Link>
                </td>
                <td>{a.customer_id}</td>
                <td>{a.address_line}</td>
                <td>{a.image_count}</td>
                <td>{a.attempt_count}</td>
                <td>{a.best_quality_score != null ? a.best_quality_score.toFixed(3) : "—"}</td>
                <td>{a.final_human_status ?? "—"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </>
  );
}
