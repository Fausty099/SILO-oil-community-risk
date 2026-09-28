<div align="center">

# SILO
### Social Impact & License to Operate Intelligence System

*An AI-powered early warning system predicting community-company conflict escalation*
*in oil-producing regions of West Africa*

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/17DyB_qtuS3nOMKyI9DwIvvh-rAO7cIPq)
[![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)](https://python.org)
[![XGBoost](https://img.shields.io/badge/Model-XGBoost-orange)](https://xgboost.readthedocs.io)
[![SHAP](https://img.shields.io/badge/Explainability-SHAP-red)](https://shap.readthedocs.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Active%20Research-brightgreen)]()

</div>

---

## The Problem Nobody Has Solved

Nigeria's Niger Delta produces over 90% of the country's oil revenue. It is also one of the most conflict-affected regions in sub-Saharan Africa.

When oil companies lose the trust of host communities — what researchers call the **Social License to Operate (SLO)** — the consequences escalate fast: protests turn to pipeline bombings, production halts, and billions in deferred output. The 2016 Niger Delta Avengers campaign alone dropped Nigeria's oil output to a **30-year low**.

The warning signs were there years before. Gas flaring rising. Protests accelerating. Fatalities spiking.

**No system existed to connect those dots.**

SILO is that system.

---

## What SILO Does

SILO fuses three publicly available data streams into a single machine learning pipeline that predicts **which state will enter conflict escalation within the next year** — before it happens.

```
NASA VIIRS Satellite Data     →  gas flare intensity & volume
Community Conflict Signals    →  protests, riots, violence, battles
Engineered Trend Features     →  lag effects, anomalies, acceleration
          ↓
    XGBoost Classifier
          ↓
SLO Erosion Risk Score (0–100%)
    + SHAP Explanation
    + Interactive Risk Map
```

**Output example:**
> *"Edo State has a 97.8% probability of entering conflict escalation — driven by rapidly accelerating conflict (+0.94 SHAP) and sustained protest activity (+0.62 SHAP)."*

That's not just a number. It's a reason. That's what makes SILO actionable.

---

## Why This Is Novel

| What Already Exists | SILO |
|---|---|
| Oil spill detection via satellite |  Widely studied |
| Community health impacts (qualitative) |  Widely studied |
| General political conflict prediction |  Widely studied |
| Resource curse economics (macro-level) |  Widely studied |
| **SLO erosion as a predictive ML target fusing environmental + conflict signals at sub-national level in West Africa** | * **Never done. This is SILO.** |

---

## Study Area

![Niger Delta Study Area](https://github.com/Fausty099/SILO-oil-community-risk/blob/cec053679099fe3555c0c9e8c91508be92513ee6/niger_delta_map.png)

**7 Niger Delta States:** Rivers · Delta · Bayelsa · Akwa Ibom · Edo · Imo · Ondo
**Study Period:** 2012 – 2023 (notebook) · 1997 – 2025 conflict panel with 2012 – 2020 site-level flare coverage (scripted pipeline)
**Total Observations:** 84 state-years (notebook) · 203 state-years 1997–2025, of which 63 carry measured site-level flare data (scripted pipeline)

---

## Key Results

> **Status note (see [`docs/REVIEW.md`](docs/REVIEW.md)).** The figures and scores in
> this section come from the original notebook, whose evaluation was not fully
> out-of-sample: the label threshold was estimated on the whole panel, the test set
> was passed to XGBoost as an `eval_set`, and risk scores were produced for rows the
> model had trained on. The reproducible pipeline in `src/silo/` re-runs the same
> question with rolling-origin validation and naive baselines; those results are in
> [`docs/REVIEW.md` §4](docs/REVIEW.md) and are substantially weaker. Treat this
> section as the v1 record and the review as the current state of evidence.

### Nigeria Gas Flaring Trend (2012–2022)
![Gas Flaring](https://github.com/Fausty099/SILO-oil-community-risk/blob/60bc0341ca7cd937971d76330c792f4a47c8e5a9/nigeria_flaring_trend.png)

Nigeria flared **132.40 BCM** of gas over the study period — declining 53% from 15.1 BCM (2012) to 7.1 BCM (2022). Yet community conflict **persisted throughout**. Environmental improvement alone does not restore social trust.

---

### Conflict & Flaring by State
![Phase 2 Summary](https://github.com/Fausty099/SILO-oil-community-risk/blob/370eaf0c0ecc5d030d11841569721bf2bd24631a/phase2_summary.png)

Rivers State leads in both cumulative flare volume and conflict events — the exact compound risk pattern SILO is designed to detect early.

---

### Comprehensive Analysis Dashboard
![EDA Overview](https://github.com/Fausty099/SILO-oil-community-risk/blob/758bd986c93a32d92f473eba5e1d8d9421e7312b/phase4_eda.png)

The conflict intensity heatmap (bottom left) reveals **Delta 2016** and **Edo** as the most chronically high-risk cells — validated by real historical events including the Niger Delta Avengers pipeline bombing campaign.

---

### SLO Erosion Detection
![SLO Labels](https://github.com/Fausty099/SILO-oil-community-risk/blob/a6d7741aa457fb9ecc1a060bbba9b845b09204d8/phase5_labels_v2.png)

**7 SLO erosion events detected** across 4 states:
- Edo: **3 events** (highest frequency)
- Bayelsa: 2 events
- Delta: 1 event — 2016 (coincides with Avengers campaign)
- Akwa Ibom: 1 event

---

### What the Model Learned
![Feature Importance](https://github.com/Fausty099/SILO-oil-community-risk/blob/a6d7741aa457fb9ecc1a060bbba9b845b09204d8/phase6_feature_importance.png)

The XGBoost model's top predictors:

| Rank | Feature | Importance |
|---|---|---|
| 1 | Conflict Events | 0.146 |
| 2 | Protests Count | 0.129 |
| 3 | Flare 3yr Baseline | 0.121 |
| 4 | Battles Count | 0.092 |
| 5 | Conflict Acceleration | 0.062 |

**Key insight:** Sustained flaring over 3+ years matters more than single-year spikes. Communities absorb acute events — they mobilize against chronic neglect.

---

### SHAP Explainability — Which Signals Drive Risk?
![SHAP Importance](https://github.com/Fausty099/SILO-oil-community-risk/blob/a6d7741aa457fb9ecc1a060bbba9b845b09204d8/phase7_shap_importance.png)

SHAP (SHapley Additive exPlanations) opens the black box. Every prediction comes with a full breakdown of which features contributed — and by how much.

**Top 3 SHAP drivers:**

| Feature | Mean \|SHAP\| | Meaning |
|---|---|---|
| Protests Count | 1.487 | Community mobilization precedes violence |
| Conflict Accel | 1.271 | Speed of escalation matters more than absolute level |
| Total Fatalities | 0.654 | Prior violence carries forward as risk memory |

![SHAP Beeswarm](https://github.com/Fausty099/SILO-oil-community-risk/blob/a6d7741aa457fb9ecc1a060bbba9b845b09204d8/phase7_shap_beeswarm.png)

Red dots = feature pushes risk UP. Blue = pushes risk DOWN. High protest activity (red, right side) is the clearest signal of future SLO erosion.

---

### Individual Prediction: Why is Edo 2019 at 97.8% Risk?
![SHAP Waterfall](https://github.com/Fausty099/SILO-oil-community-risk/blob/a6d7741aa457fb9ecc1a060bbba9b845b09204d8/phase7_shap_waterfall.png)

```
Baseline (average risk)      →  0.879
+ Conflict Acceleration      →  +0.94
+ Protests Count             →  +0.62
+ Total Fatalities           →  +0.46
+ Conflict YoY               →  +0.30
+ Violence Ratio             →  +0.27
+ Other features             →  +0.28
                                ──────
Final score                  →  3.818  →  97.8% risk
```

### High Risk vs Low Risk — Side by Side
![SHAP Comparison](https://github.com/Fausty099/SILO-oil-community-risk/blob/a6d7741aa457fb9ecc1a060bbba9b845b09204d8/phase7_shap_comparison.png)

**Edo 2019 (97.8%)** vs **Ondo 2016 (0.7%)** — same model, same features, opposite outcomes. SHAP shows exactly what separates them.

---

## Model Architecture

```python
# Temporal split — never train on future to predict past
Train: 2012–2019  (42 observations, 5 positive labels)
Test:  2020–2023  (21 observations)

# Class imbalance handled explicitly
scale_pos_weight = 11.6  # negative:positive ratio

XGBClassifier(
    n_estimators     = 200,
    max_depth        = 3,
    learning_rate    = 0.05,
    subsample        = 0.8,
    colsample_bytree = 0.8,
    scale_pos_weight = 11.6
)
```

**SLO Erosion Label Definition:**
> A state receives label = 1 if conflict events in year t+1 exceed the **60th percentile threshold (70.0 events)** AND exceed **1.2× the state's own 3-year rolling baseline** — capturing sudden surges above historical norms, not chronically tense states.

---

## Highest Risk State-Years

| State | Year | Risk Score | Confirmed Erosion |
|---|---|---|---|
| Edo | 2019 | **97.8%** | Confirmed |
| Akwa Ibom | 2015 | 95.5% | Confirmed |
| Delta | 2016 | 94.8% | Confirmed |
| Bayelsa | 2019 | 94.5% | Confirmed |
| Edo | 2014 | 94.3% | Confirmed |

All confirmed SLO erosion cases received risk scores above **94%**.

---

## Reproducible Pipeline (`src/silo/`)

The notebook is Colab-only and reads hand-uploaded files. `src/silo/` is a scripted
replacement that downloads its own inputs and can be re-run end to end:

```bash
pip install -r requirements.txt
python -m silo.acquire     # ACLED (HDX API), VIIRS flare sites (EOG), GADM v4.1
python -m silo.pipeline    # panel -> features -> labels -> benchmark -> figures
pytest                     # 9 tests, incl. a feature-leakage regression test
```

| Module | Role |
|---|---|
| `acquire.py` | Resolves current HDX resource URLs and caches raw inputs |
| `panel.py` | State x year panel; **point-in-polygon join of individual VIIRS flare sites** onto GADM polygons (replaces fixed per-state flare shares) |
| `features.py` | Lag / rolling / anomaly features, all computable at time *t* |
| `labels.py` | SLO-erosion proxy; percentile threshold fitted **inside the training fold** |
| `evaluate.py` | Naive persistence + historical base-rate baselines, 5-model benchmark, rolling-origin CV, Brier score, reliability curves |
| `pipeline.py` | Runs everything, writes `outputs/tables/`, `outputs/figures/`, `models/` |

Two specifications are reported: conflict + satellite flaring (2012-2019, the years
with site-level VIIRS coverage) and conflict history only (2000-2024, far more
observations).

---

## Data Sources

| Dataset | Source | Cost |
|---|---|---|
| Gas Flaring Volume | World Bank GGFR / NASA VIIRS | Free |
| Conflict Events | ACLED / HDX Nigeria | Free |
| State Boundaries | GADM v4.1 | Free |

**No proprietary data. No paywalls. Fully reproducible.**

---

## Project Structure

```
SILO-oil-community-risk/
│
├── SILO_project.ipynb          ← Original research notebook (Google Colab)
├── src/silo/                   ← Reproducible pipeline (acquire, panel, features,
│                                  labels, evaluate, pipeline)
├── tests/                      ← pytest suite incl. leakage regression test
├── docs/REVIEW.md              ← Code + proposal review, baselines & calibration results
│
├── data/
│   ├── raw/
│   │   ├── viirs/              ← Gas flaring data
│   │   ├── acled/              ← Conflict event data
│   │   └── boundaries/         ← Nigeria state shapefiles
│   └── processed/
│       ├── silo_panel.csv      ← Master state × year panel (84 rows)
│       ├── silo_features.csv   ← 24 engineered features
│       └── silo_model_ready.csv← Labelled dataset
│
├── models/
│   └── silo_xgboost_v1.json   ← Trained XGBoost model
│
└── outputs/
    ├── SILO maps %26 figures/                ← 12 generated charts
    └── silo_risk_map.html      ← Interactive choropleth map
```

---

## Pipeline

```
Phase 0  │  Environment Setup
Phase 1  │  Data Acquisition  (NASA VIIRS + ACLED + GADM)
Phase 2  │  Spatial Joining   (point data → state polygons)
Phase 3  │  Feature Engineering  (24 features across 3 streams)
Phase 4  │  Exploratory Data Analysis  (6 visualisations)
Phase 5  │  Label Construction  (SLO erosion proxy target)
Phase 6  │  Model Training  (XGBoost + temporal split)      
Phase 7  │  SHAP Explainability  (4 explanation charts)     
Phase 8  │  Interactive Risk Map  (Folium choropleth)       
```

---

## How to Run

**Option 1 — Google Colab (Recommended — zero setup)**

Click the badge at the top. Phase 0 installs everything automatically.

**Option 2 — Local (scripted pipeline, no manual uploads)**

```bash
git clone https://github.com/Fausty099/SILO-oil-community-risk.git
cd SILO-oil-community-risk
pip install -r requirements.txt
python -m silo.acquire && python -m silo.pipeline
```

**Option 3 — Local notebook**

```bash
pip install -r requirements.txt
jupyter notebook SILO_project.ipynb   # requires the ACLED/PRIMAP files uploaded manually
```

---

## Roadmap

- [x] Phase 0–4: Full data pipeline, spatial analysis, EDA
- [x] Phase 5: SLO erosion label construction
- [x] Phase 6: XGBoost model training
- [x] Phase 7: SHAP explainability
- [x] Phase 8: Interactive risk map
- [ ] Research paper submission → *Resources Policy* (Elsevier)
- [ ] Version 2: LGA-level resolution + FAAC revenue data
- [ ] Version 3: Extension to Ghana Jubilee Field
- [ ] Version 4: East Africa (EACOP pipeline context)

---

## Who This Is For

**Oil companies** → Early signal to course-correct community engagement before SLO collapse

**Policymakers & regulators** → Evidence base for proactive intervention by state

**ESG investors** → Systematic community risk due diligence beyond qualitative assessments

**Affected communities** → A data voice that surfaces conditions of harm before they become irreversible

---

## Author

<div align="center">

**Faustina Afua Afriyie**
BSc Data Science & Analytics — Level 300
Ghana Communication Technology University (GCTU), Accra, Ghana

[![GitHub](https://img.shields.io/badge/GitHub-Fausty099-black?logo=github)](https://github.com/Fausty099)

</div>

---

## License

MIT License — free to use with attribution.

---

## Acknowledgements

- World Bank Global Gas Flaring Reduction Partnership (GGFR)
- Armed Conflict Location & Event Data Project (ACLED)
- GADM Database of Global Administrative Areas
- NASA Earth Observation Group — VIIRS Nightfire

---

<div align="center">

*SILO is an independent academic research project.*
*Not affiliated with any oil company, government body, or political organization.*



</div>
