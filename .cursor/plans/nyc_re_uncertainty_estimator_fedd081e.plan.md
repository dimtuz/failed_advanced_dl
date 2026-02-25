---
name: NYC RE Uncertainty Estimator
overview: Build an automated NYC Real Estate Uncertainty Estimator in a single Jupyter notebook (`src/notebook.ipynb`) that ingests live NYC Rolling Sales data, engineers LLM-based features via Gemini, trains a two-headed aleatoric+epistemic DNN adapted from M1_3, performs SHAP analysis, and optionally automates via GitHub Actions + GCP.
todos:
  - id: phase1-data
    content: "Phase 1: Download and clean NYC Rolling Sales CSV (price, sqft, age, dedup)"
    status: completed
  - id: phase2-llm
    content: "Phase 2: Gemini LLM feature engineering — sub_region + affluence_score per neighborhood"
    status: completed
  - id: phase3-model
    content: "Phase 3: Build and train two-headed DNN (mu + sigma heads, NLL loss) adapted from M1_3"
    status: completed
  - id: phase4-uncertainty
    content: "Phase 4: Compute aleatoric + epistemic uncertainty, interval coverage, outlier inspection"
    status: completed
  - id: phase4-shap
    content: "Phase 4: SHAP analysis — beeswarm, dependence, waterfall, quadrant plots for all three heads"
    status: completed
  - id: phase5-automation
    content: "Phase 5 (stretch): requirements.txt, GitHub Actions YAML, GCS upload script"
    status: completed
isProject: false
---

# NYC Real Estate Uncertainty Estimator — Implementation Plan

## Reference Material

- Template: `[Context/M1_3_DNNs_SHAP_Aleatoric.ipynb](Context/M1_3_DNNs_SHAP_Aleatoric.ipynb)` — dual-head DNN, NLL loss, MC Dropout, SHAP
- Task: `[Context/initial_task.txt](Context/initial_task.txt)` — requires loss curves, SHAP, interval analysis, outlier inspection

## Output: `src/notebook.ipynb`

All work lives in one notebook with clearly labeled sections.

---

## Phase 1 — Live Data Ingestion

- Download NYC Citywide Rolling Calendar Sales CSV from NYC Open Data API via `requests`
  - URL: `https://data.cityofnewyork.us/api/views/usep-8jbt/rows.csv?accessType=DOWNLOAD`
- Clean `SALE PRICE`: strip `$`/commas, cast to float, drop rows where price ≤ 0 (ownership transfers)
- Clean `GROSS SQUARE FEET` and `LAND SQUARE FEET`: coerce to numeric, drop zeros
- Derive `BUILDING AGE` = current year − `YEAR BUILT`
- Drop duplicates (as required by the task spec)
- Output: clean `pd.DataFrame` with ~50k+ rows

## Phase 2 — LLM Feature Engineering (Gemini)

- Collect unique `NEIGHBORHOOD` strings from the dataframe
- Call `google-generativeai` (Gemini Flash) with a structured prompt asking it to return JSON mapping each neighborhood to:
  - `sub_region` (one of 10 broader NYC sub-regions)
  - `affluence_score` (integer 1–10)
- Parse JSON response and map back to the main dataframe as two new columns
- Cache result to avoid repeat API calls during development

Key snippet pattern (from M1_3, adapted):

```python
import google.generativeai as genai
model = genai.GenerativeModel("gemini-1.5-flash")
response = model.generate_content(prompt)
neighborhood_map = json.loads(response.text)
```

## Phase 3 — Two-Headed Model (Adapted from M1_3)

### Features (input to model)

- `GROSS SQUARE FEET` (scaled)
- `BUILDING AGE` (scaled)
- `BOROUGH` (one-hot or label encoded)
- `affluence_score` (from LLM)
- `sub_region` (label encoded)
- Optionally: `RESIDENTIAL UNITS`, `TOTAL UNITS`

### Architecture (mirrors M1_3 `make_combined_model`)

```
Input(shape=(N_features,))
→ Dense(128, relu) → BatchNorm → Dropout(0.2, training=True)
→ Dense(64, relu)  → BatchNorm
→ Head 1: Dense(1, name="mu")                    — predicted log price
→ Head 2: Dense(1, activation='softplus', name="sigma_sq") — aleatoric variance
→ Concatenate([mu, sigma_sq])
```

### Loss

Gaussian NLL (identical to M1_3 `aleatoric_loss`):

```
L = 0.5*log(σ²+ε) + 0.5*(y−μ)²/(σ²+ε)
```

Target: `log(SALE PRICE)` (log-transforms heavy-tailed price distribution)

### Training

- Adam lr=0.001, up to 300 epochs, EarlyStopping(patience=15)
- Plot train/val NLL loss curves
- Evaluate: MAE, RMSE, R² on point estimate (mu head)
- Baseline comparison: `sklearn` `LinearRegression` or `DummyRegressor`

## Phase 4 — Uncertainty & SHAP Analysis

### Epistemic Uncertainty

- T=50 stochastic passes with `model(X, training=True)`
- `epistemic_std = all_mu.std(axis=0)` (in log-price, convert back to dollars)

### Aleatoric Uncertainty

- Single pass `model(X, training=False)`, take `sigma_sq` head
- `aleatoric_std = sqrt(sigma_sq) * price_scale`

### Analyses (matching M1_3 structure)

- Actual vs. Predicted scatter (colored by total uncertainty)
- Uncertainty histograms: epistemic vs. aleatoric side-by-side
- 95% interval coverage check (Gaussian + empirical)
- **Outlier Inspection**: flag rows where `SALE PRICE` is high AND aleatoric uncertainty is high → "unpredictable luxury" cluster

### SHAP

Three `shap.KernelExplainer` wrappers (same pattern as M1_3):

- `predict_mean_final` → price
- `predict_epistemic_final` → epistemic std (50 passes)
- `predict_aleatoric_final` → aleatoric std (single pass)

Analyses:

- Beeswarm summary plots for all three
- Dependence plots for `affluence_score` and `GROSS SQUARE FEET`
- Waterfall plots for the most uncertain houses
- Four-quadrant SHAP magnitude × uncertainty magnitude scatter

## Phase 5 — GitHub Actions + GCP Automation (Stretch)

### Files to create

- `requirements.txt` — pinned deps (tensorflow, shap, google-generativeai, etc.)
- `.github/workflows/nightly.yml` — scheduled GitHub Action:
  - Runs on `cron: '0 6 * * *'` (nightly)
  - Converts notebook to `.py` with `nbconvert`, runs it
  - Authenticates to GCP via `GOOGLE_APPLICATION_CREDENTIALS` secret
  - Uploads daily uncertainty report CSV + plots to a GCS bucket via `google-cloud-storage`
- GCP: Service account with `Storage Object Creator` role

### GH Action skeleton

```yaml
on:
  schedule:
    - cron: '0 6 * * *'
jobs:
  run-model:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
      - run: pip install -r requirements.txt
      - run: jupyter nbconvert --to script src/notebook.ipynb && python src/notebook.py
      - run: python scripts/upload_to_gcs.py
        env:
          GCS_BUCKET: ${{ secrets.GCS_BUCKET }}
          GOOGLE_APPLICATION_CREDENTIALS_JSON: ${{ secrets.GCP_SA_KEY }}
```

---

## File Structure

```
advanced_dl_project/
├── src/
│   └── notebook.ipynb         ← main deliverable
├── scripts/
│   └── upload_to_gcs.py       ← GCS upload helper (Phase 5)
├── requirements.txt
└── .github/
    └── workflows/
        └── nightly.yml        ← GitHub Actions (Phase 5)
```

