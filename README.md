# Record Linkage Dedup

> **Domain:** Clinical Decision Support & Biomedical Computing

<div align="center">

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-3776AB.svg?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688.svg?logo=fastapi&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg?logo=docker&logoColor=white)

</div>

---

## Overview

Record Linkage Deduplicator identifies likely duplicate patient records across one or more datasets using the **Fellegi-Sunter (1969) probabilistic record linkage model**, with a Winkler-style EM step to estimate agreement/disagreement weights and Jaro-Winkler string similarity for fuzzy comparison of name fields.

---

## Features

- **Fellegi-Sunter probabilistic model** with m/u probability estimation
- **Jaro-Winkler fuzzy string matching** for name fields
- **Blocking** (birth year + first-name initial) to avoid O(n^2) comparisons
- **EM algorithm** for automatic m/u calibration from data
- **PHI outbound guard** to prevent sensitive data leakage
- **HMAC-SHA256 tamper-evident audit trail**
- **FastAPI REST API** for remote processing
- **Prometheus metrics** endpoint

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Usage

### Command Line Interface

**Deduplicate a single dataset:**
```bash
python linkage_dedup.py data/dataset_a.csv
```

**Link two datasets:**
```bash
python linkage_dedup.py data/dataset_a.csv data/dataset_b.csv --output report.csv
```

**Use EM to estimate m/u probabilities:**
```bash
python linkage_dedup.py data/dataset_a.csv --em
```

**Run with custom field specs:**
```bash
python linkage_dedup.py data/dataset_a.csv --fields first_name:jw,last_name:jw,dob:near
```

### Enterprise CLI (agents framework)

```bash
# Run audit
python cli.py audit --task-id TASK-001 --primary 28.5

# Batch process
python cli.py batch -i data/dataset_a.csv -o results.csv

# Launch API server
python cli.py serve --host 127.0.0.1 --port 8000

# Verify audit trail
python cli.py verify-audit
```

### REST API

```bash
python cli.py serve
```

Endpoints:
- `GET /health` - Health check
- `GET /metrics` - Prometheus metrics
- `POST /api/audit` - Submit task for evaluation
- `POST /api/chat` - Supervisory chat
- `GET /api/audit/logs` - Audit trail

---

## Input Data Schema

CSV files with the following columns:

| Field | Description | Requirement |
|:------|:------------|:------------|
| `patient_id` | Unique record identifier | Required |
| `first_name` | Patient first name | Optional |
| `last_name` | Patient last name | Optional |
| `dob` | Date of birth (YYYY-MM-DD) | Optional |

---

## Testing

```bash
pytest -v
```

Run simulation benchmark:

```bash
python simulator.py 1000
```

---

## Security

- **Zero-PHI Outbound Guard:** AST and regex inspection blocking SSNs, MRNs, phone numbers
- **HMAC-SHA256 Audit Trail:** Chained, cryptographically signed logs
- **Path traversal protection** on file operations

Set a strong audit secret key via environment variable:
```bash
export AUDIT_SECRET_KEY=$(python -c "import secrets; print(secrets.token_hex(32))")
```

---

## Docker Deployment

```bash
docker-compose up --build
```

Or manually:

```bash
docker build -t record-linkage-dedup .
docker run -p 8000:8000 -e AUDIT_SECRET_KEY=your-secret-key record-linkage-dedup
```

---

## Project Structure

```
record-linkage-dedup/
├── record_linkage/      # Core linkage package
│   ├── similarity.py    # Jaro/Jaro-Winkler
│   ├── model.py         # Fellegi-Sunter model
│   ├── blocking.py      # Blocking key construction
│   ├── linker.py        # Pipeline orchestration
│   └── cli.py           # Linkage CLI
├── agents/              # Enterprise agent framework
│   ├── base.py          # PHI guard, audit trail
│   ├── models.py        # Pydantic schemas
│   ├── supervisor.py    # Orchestrator
│   ├── workers.py       # Specialized agents
│   └── api.py           # FastAPI endpoints
├── tests/               # Pytest test suite
├── data/                # Sample datasets
├── cli.py               # Enterprise CLI entry point
├── linkage_dedup.py     # Linkage CLI entry point
├── enrichment.py        # Feature enrichment suite
├── simulator.py         # Load testing simulator
└── requirements.txt     # Python dependencies
```
