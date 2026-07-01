# Human Activity Recognition (HAR) from Wearable Sensor Data

Classifying physical activities (walking, sitting, standing, lying, walking
upstairs/downstairs) from smartphone accelerometer and gyroscope signals.

**Author:** Abel Raphel Pulikottil
**Status:** 🚧 Actively building (portfolio project)

## Why this project

Human Activity Recognition is the core technology behind **fall detection,
elderly-care monitoring, and rehabilitation tracking** — a key building block
in assistive health systems and healthy-ageing research.

This project is built following the **CRISP-DM** methodology, one phase at a
time, with an emphasis on understanding and explaining every decision — not
just producing numbers.

## Dataset

[UCI Human Activity Recognition Using Smartphones](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones)
— 30 subjects, 6 activities, recorded with a waist-mounted smartphone.
*(Not committed to this repo; see `data/` and `.gitignore`.)*

## Project structure

```
├── data/          # Raw dataset (git-ignored)
├── notebooks/     # Analysis notebooks, one per CRISP-DM phase
├── src/           # Reusable Python modules (extracted later)
└── reports/       # Figures, confusion matrices, findings
```

## Roadmap (CRISP-DM)

- [x] Phase 1 — Business Understanding ([write-up](reports/01_business_understanding.md))
- [x] Phase 2 — Data Understanding ([write-up](reports/02_data_understanding.md))
- [x] Phase 3 — Data Preparation ([write-up](reports/03_data_preparation.md))
- [x] Phase 4 — Modeling — [baseline: RF 92.87%](reports/04_modeling_baseline.md) + [CNN 93.35%](reports/05_modeling_deep.md)
- [x] Phase 5 — Evaluation ([summary](reports/06_evaluation.md)) — best model **ensemble 94.9%**, subject-wise CV 0.938 ± 0.022
- [ ] Phase 6 — Deployment / Communication

## Reproducing this project

```bash
python -m venv .venv
.venv\Scripts\activate      # Windows
pip install -r requirements.txt
```
### Step 0 — Project setup (CRISP-DM: pre-work)
- Created repo structure: `data/` (git-ignored), `notebooks/`, `src/`, `reports/figures/`.
- Wrote `.gitignore` to exclude the raw dataset, virtual environment, and model checkpoints.
- Added `README.md` (problem framing + CRISP-DM roadmap) and `requirements.txt`.
- Initialized Git and made the first commit.