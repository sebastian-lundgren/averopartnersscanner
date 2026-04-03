import TrainingPanel from "./TrainingPanel";
import { apiGet } from "@/lib/api";

type Mv = {
  id: number;
  version_tag: string;
  description: string | null;
  is_active: boolean;
  weights_path: string | null;
  metrics_json: Record<string, number> | null;
  training_config_json: Record<string, unknown> | null;
  train_image_count: number | null;
};

export default async function ModelsPage() {
  let versions: Mv[] = [];
  let compare: { versions: unknown[] } | null = null;
  let err = "";
  try {
    versions = await apiGet<Mv[]>("/api/model-versions");
    compare = await apiGet<{ versions: unknown[] }>("/api/model-versions/compare-summary");
  } catch (e) {
    err = e instanceof Error ? e.message : "Feil";
  }

  return (
    <>
      <h1>Modellversjoner</h1>
      {err && <p className="muted">{err}</p>}
      <TrainingPanel versions={versions} />
      <div className="card">
        <h3>Registrerte versjoner</h3>
        <ul>
          {versions.map((v) => (
            <li key={v.id}>
              <strong>{v.version_tag}</strong>
              {v.is_active ? " (aktiv for prediksjoner i DB)" : ""} — {v.description}
              {v.metrics_json && (
                <span className="muted" style={{ fontSize: 12 }}>
                  {" "}
                  metrics: {JSON.stringify(v.metrics_json)}
                </span>
              )}
            </li>
          ))}
        </ul>
      </div>
      <div className="card">
        <h3>Sammenligning (enkel)</h3>
        <pre style={{ fontSize: 12, overflow: "auto" }}>
          {JSON.stringify(compare?.versions ?? [], null, 2)}
        </pre>
      </div>
    </>
  );
}
