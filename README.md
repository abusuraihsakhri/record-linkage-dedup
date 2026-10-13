# Record Linkage Dedup

Probabilistic record linkage and candidate-duplicate identification for CSV data. The repository includes a browser application, a configurable Python pipeline, and a separate demonstration audit service.

## Browser application

Open `web/index.html` through a local HTTP server, or the GitHub Pages deployment when enabled:

```bash
python -m http.server 8080
# Browse to http://localhost:8080/web/
```

Choose one or two CSV files or load the sample dataset, select a name similarity threshold, and choose **Find candidate duplicates**. Review the scored candidate pairs and download the detailed CSV report. The browser application operates entirely in the tab: files and results are not uploaded to a server.

The browser implementation supports `first_name`, `last_name`, and `dob` (ISO `YYYY-MM-DD`), with birth-year/first-initial blocking and fixed Fellegi–Sunter m/u probabilities of 0.9/0.1. It compares candidates both within and across supplied files. It does not perform EM estimation or support custom field selection. To limit browser resource usage it accepts at most 20,000 rows, 5 MB per file, and 100,000 candidate pairs; larger or more specialized tasks should use the Python CLI.

## Python command line

Use Python 3.10–3.12:

```bash
python -m pip install -r requirements.txt
python linkage_dedup.py data/dataset_a.csv data/dataset_b.csv --output linkage_report.csv
python linkage_dedup.py data/dataset_a.csv --em
python linkage_dedup.py data/dataset_a.csv --fields first_name:jw,last_name:jw,dob:near
python linkage_dedup.py --help
```

The Python engine provides Jaro and Jaro–Winkler fuzzy comparisons, configurable blocking, field selection, and optional unsupervised EM estimation of Fellegi–Sunter probabilities. Default inferred fields exclude record identifiers such as `patient_id` and `mrn` (specify them explicitly if desired). An empty candidate set writes a header-only CSV. A blocked-out pair is **not** proof of nonidentity; blocking can miss true matches.

For the sample data, the report lists candidate record pairs with scores and `match`, `possible_match`, or `non_match` classifications. These classes are heuristic and require external validation before record merging or clinical use.

### CSV input

The browser requires columns `first_name,last_name,dob` with DOB in `YYYY-MM-DD` format. The Python CLI can infer common alternatives (or accept explicit `--fields` and `--blocking`). Records from all input files are combined; both within-file and cross-file pairs may be considered. Missing blocking-key components are skipped.

## Separate audit demonstration

An auxiliary service in `agents/` implements numeric rule evaluation and a mock supervisory query workflow; it is **not** the record-linkage engine. To try it locally:

```bash
python cli.py audit --task-id TASK-001 --primary 28.5
python cli.py verify-audit
python cli.py serve --host 127.0.0.1 --port 8000
```

The server exposes `GET /health`, `GET /metrics` (JSON, not Prometheus exposition), `POST /api/audit`, `POST /api/chat`, and `GET /api/audit/logs`. The audit log uses per-process HMAC-SHA256 chain verification. It is in-memory and is **not** durable across restarts. Set `AUDIT_SECRET_KEY` to a high-entropy value to control the signing key. The mock chat endpoint does not connect to an external language model.

## Privacy and security

Browser CSV processing makes no outgoing data requests. CSV report export escapes spreadsheet-formula-like values. The Python CLI reads local paths and can output sensitive matching data; handle these files according to your local privacy obligations. The audit service includes pattern-based detection of selected sensitive strings, but it cannot guarantee detection of all protected health information, HIPAA de-identification, or clinical compliance. The demonstration API does not implement authentication: **do not expose it on a public network or send identifiable patient data to it**.

## Development and testing

```bash
python -m pip install -r requirements.txt
python -m pytest -q
python -m compileall -q agents record_linkage cli.py linkage_dedup.py simulator.py enrichment.py
node --test web/*.test.mjs
```

The browser application uses native JavaScript modules, HTML, and CSS without framework dependencies. The Python CLI requires NumPy and pandas; the auxiliary API uses FastAPI and Pydantic. The browser build targets current versions of Chromium, Firefox, and Safari, though cross-browser manual testing is still recommended.

GitHub Actions runs Python tests on 3.10, 3.11, and 3.12 and tests the browser matching engine with Node. A separate workflow deploys the `web/` directory to GitHub Pages when Pages is configured with GitHub Actions as the build source.

## License

No license file is currently included in the repository. Consult the repository owner for reuse permissions.
