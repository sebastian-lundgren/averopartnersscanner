# Alarmskilt QC — lokal MVP

Review- og treningsverktøy for **visuell kvalitetskontroll av alarmskilt** i et **autorisert testmiljø**. Bygget for:

- egne kundeadresser med samtykke/avtale,
- intern QC av egne installasjoner,
- manuell gjennomgang av brukeropplastede eller samtykkebaserte bilder,
- læring fra rettinger over tid.

**Dette systemet er ikke ment for:** massekartlegging av private hjem, adresselister over sårbarhetsstatus, eller kategorien «har ikke alarm» / «ingen skilt» som endelig utfall. Kun disse tre statusene brukes:

| Status | Beskrivelse |
|--------|-------------|
| `skilt_funnet` | Modellen/heuristikken foreslår at et skilt kan være synlig (krever ofte likevel menneskelig QA). |
| `uklart` | Unknown-first ved tvil (lys, vinkel, avstand, skarphet, …). |
| `trenger_manuell` | Skal alltid til menneske — ingen sikker maskinkonklusjon. |

---

## Prosjektstruktur

```
Streetview scanner/
├── README.md
├── .gitignore
├── backend/
│   ├── .env.example
│   ├── requirements.txt
│   ├── app/
│   │   ├── main.py              # FastAPI-app, CORS, routere
│   │   ├── config.py
│   │   ├── database.py
│   │   ├── models.py            # SQLAlchemy: ImageAsset, AddressRecord, Prediction, …
│   │   ├── schemas.py           # Pydantic API-skjemaer
│   │   ├── seed.py              # Modellversjoner + demo-adresse + syntetiske bilder
│   │   ├── routers/             # images, addresses, reviews, dashboard, export, …
│   │   └── services/            # prediction, quality, evidence, best_view, active_learning, settings
│   ├── scripts/
│   │   └── retrain_placeholder.py   # Eksporter training_examples → JSONL (Fase 3-stub)
│   └── data/                    # Opprettes lokalt: app.db, uploads/, evidence/
└── frontend/
    ├── package.json
    ├── next.config.ts
    ├── .env.local.example
    ├── app/                     # Next.js App Router: dashboard, upload, library, review, …
    ├── components/Nav.tsx
    └── lib/api.ts, lib/constants.ts
```

### Datamodeller (implementert)

| Modell | Hovedfelt |
|--------|-----------|
| **ImageAsset** | `stored_path`, `evidence_crop_path`, `address_id`, `quality_score`, `is_temporary_candidate`, `is_primary_for_address`, `discard_reason` |
| **AddressRecord** | `customer_id`, `address_line`, `attempt_count`, `best_quality_score`, `selected_image_id`, `final_human_status`, `selection_metadata_json` |
| **Prediction** | `predicted_status`, `confidence`, `bbox_json`, `rationale`, `needs_review`, `priority_score`, `review_completed`, `model_version_id` |
| **ReviewDecision** | `final_status`, `was_override`, `comment`, `error_type` |
| **ModelVersion** | `version_tag`, `is_active`, `metrics_json` |
| **TrainingExample** | rettelser: `human_status`, `original_model_guess`, `model_version_tag`, `confidence_at_time`, `error_type`, `evidence_crop_path` |
| **TrainingLibraryEntry** | eksempelbibliotek med `category` + `tags_json` |
| **AppSetting** | JSON-innstillinger (terskler) |

`ExportRecord` er ikke lagret som egen tabell i MVP (eksport genereres on-demand). Du kan legge til logg-tabell senere om du trenger sporbarhet per eksport.

---

## Installasjon og oppstart

### 1. Backend (Python 3.11+)

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env        # valgfritt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Ved første oppstart opprettes SQLite (`data/app.db`), standard terskler og **demo-data** (to syntetiske PNG-bilder + prediksjoner på demo-adresse).

API-dokumentasjon: [http://localhost:8000/docs](http://localhost:8000/docs)

### 2. Frontend (Next.js 15)

```bash
cd frontend
cp .env.local.example .env.local
npm install
npm run dev
```

Åpne [http://localhost:3000](http://localhost:3000). `NEXT_PUBLIC_API_URL` skal peke på backend (standard `http://localhost:8000`).

### 3. Fase 3 — eksporter treningsmanifest

```bash
cd backend
source .venv/bin/activate
python scripts/retrain_placeholder.py
```

Skriver `data/export_training_manifest.jsonl` fra tabellen `training_examples`.

---

## Viktige API-endepunkter

| Metode | Sti | Formål |
|--------|-----|--------|
| POST | `/api/images/upload` | Enkelt/batch-opplasting, valgfri `address_id` / kunde+adresse |
| GET | `/api/images/library` | Bildebibliotek |
| GET | `/api/reviews/queue` | Review-kø (sortert etter `priority_score` — aktiv læring, enkel) |
| POST | `/api/reviews/{id}/submit` | Godkjenn / overstyre + feiltype + kommentar |
| GET | `/api/dashboard/stats` | Dashboard-tall |
| GET | `/api/export/csv` / `/api/export/xlsx` | Eksport |
| GET/PUT | `/api/settings/thresholds` | Justerbare terskler |
| POST | `/api/addresses/{id}/best-view` | Best view blant `is_temporary_candidate`-bilder |
| GET | `/api/files/image/{id}/original` | Bilde til UI |

---

## ML / prediksjon (MVP)

- **heuristic-v1**: OpenCV + bildekvalitet (Laplacian, eksponering, kontrast) og svake kant-heuristikker i fasaderegion.
- **Unknown-first**: ved dårlig sikt returneres `uklart` eller `trenger_manuell`, aldri «ingen alarm».
- Evidensutsnitt lagres når en bounding box kan utledes (normaliserte koordinater i `bbox_json`).

---

## Review — tastatursnarveier

På **Review-kø** (`/review`):

- `1` / `2` / `3` — velg status skilt funnet / uklart / trenger manuell  
- `A` — godkjenn modellens forslag  
- `Enter` — send med valgt status  
- `P` / `N` — forrige / neste i listen (når flere i kø)

---

## Hva som gjenstår (anbefalt rekkefølge)

**Fase 2 (allerede delvis i kode):** finere evidens (bedre deteksjon), rikere aktiv læring (embeddings-likhet med tidligere feil), dedikert «motstridende vurderinger»-view.

**Fase 3:** ekte retreningsskript (PyTorch/YOLO/CLIP e.l.), valideringssett, evalueringsrapport (presisjon/recall kun på de tre klassene + menneskelig baseline).

**Produksjon:** autentisering, backup, ikke lokal SQLite, revisjonsspor for alle feltendringer, policy for sletting av midlertidige kandidatbilder.

---

## Lisens / bruk

Kun bruk i samsvar med gjeldende personvern, avtaler og samtykke. Utviklere og operatører er ansvarlige for å ikke bruke verktøyet til ulovlig kartlegging.
