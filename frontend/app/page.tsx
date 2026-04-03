import { apiGet } from "@/lib/api";

type Stats = {
  total_images: number;
  total_predictions: number;
  count_skilt_funnet: number;
  count_uklart: number;
  count_trenger_manuell: number;
  overrides_count: number;
  pending_review: number;
  error_rate_last_7d: number | null;
  by_model_version: { version: string; predictions: number; overrides: number }[];
};

export default async function DashboardPage() {
  let stats: Stats | null = null;
  let err = "";
  try {
    stats = await apiGet<Stats>("/api/dashboard/stats");
  } catch (e) {
    err = e instanceof Error ? e.message : "Kunne ikke hente statistikk";
  }

  return (
    <>
      <div className="banner">
        Dette verktøyet er for <strong>autoriserte</strong> use cases (samtykke/avtale, intern QC,
        manuelt opplastede bilder). Det bygges ikke for massekartlegging eller &quot;ingen
        alarm&quot;-lister. Kun statusene: Skilt funnet, Uklart, Trenger manuell vurdering.
      </div>
      <h1>Dashboard</h1>
      {err && <p className="muted">{err} — start backend (se README).</p>}
      {stats && (
        <div className="grid2">
          <div className="card">
            <h3>Bilder og kø</h3>
            <p>Totalt bilder: {stats.total_images}</p>
            <p>Prediksjoner: {stats.total_predictions}</p>
            <p>
              <strong>Venter review:</strong> {stats.pending_review}
            </p>
          </div>
          <div className="card">
            <h3>Endelige vurderinger (etter review)</h3>
            <p>Skilt funnet: {stats.count_skilt_funnet}</p>
            <p>Uklart: {stats.count_uklart}</p>
            <p>Trenger manuell vurdering: {stats.count_trenger_manuell}</p>
            <p>Manuelle overstyringer: {stats.overrides_count}</p>
            <p>
              Feilrate siste 7 d:{" "}
              {stats.error_rate_last_7d != null
                ? `${(stats.error_rate_last_7d * 100).toFixed(1)} %`
                : "—"}
            </p>
          </div>
          <div className="card" style={{ gridColumn: "1 / -1" }}>
            <h3>Per modellversjon</h3>
            <table style={{ width: "100%", borderCollapse: "collapse" }}>
              <thead>
                <tr style={{ textAlign: "left", borderBottom: "1px solid var(--border)" }}>
                  <th>Versjon</th>
                  <th>Prediksjoner</th>
                  <th>Overstyringer</th>
                </tr>
              </thead>
              <tbody>
                {stats.by_model_version.map((r) => (
                  <tr key={r.version}>
                    <td>{r.version}</td>
                    <td>{r.predictions}</td>
                    <td>{r.overrides}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </>
  );
}
