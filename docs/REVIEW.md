# SILO — code and proposal review

Scope: `SILO_project.ipynb`, `README.md` and `SILO_Research_Proposal.pdf`.
Everything below that reports a number was produced by the new pipeline in
`src/silo/` on public data; nothing is carried over from the notebook's own output.

---

## 1. The headline problem: reported performance was not out-of-sample

Three separate issues in the notebook all inflate the reported scores in the same
direction, and they compound:

1. **The label threshold was fitted on the whole panel.**
   `p60 = df_model['future_conflict'].quantile(0.60)` uses every year, including the
   test years. The definition of "erosion" therefore depends on outcomes the model is
   later evaluated on. The threshold is an estimated parameter and belongs inside the
   training fold.
2. **XGBoost was trained with the test set as its evaluation set.**
   `eval_set=[(X_test, y_test)]` turns the test set into a tuning signal.
3. **Risk scores were produced for all rows from a model trained on most of them.**
   The published risk map therefore largely reports in-sample fit.

With those three fixed and evaluation moved to a rolling-origin (expanding-window)
scheme, the picture changes completely — see §4.

## 2. Flare allocation was assumed, not measured

The notebook distributed the *national* flare total across states with hard-coded
shares:

```python
state_flare_share = {'Rivers': 0.28, 'Delta': 0.25, 'Bayelsa': 0.20,
                     'Akwa Ibom': 0.12, 'Edo': 0.07, 'Imo': 0.05, 'Ondo': 0.03}
```

Because these shares are constant in time, every state's flare series is a rescaled
copy of the national series. All cross-state variation in flaring is therefore an
artefact of the constants, and any "flaring predicts conflict" finding is really
"national flaring trend + a state constant".

The new pipeline replaces this with a point-in-polygon join of **individual VIIRS
flare sites** (Earth Observation Group, 2012–2020) onto GADM v4.1 state polygons.
Measured onshore shares over 2012–2020:

| State | Assumed share | Measured share |
|---|---|---|
| Delta | 0.25 | **0.363** |
| Rivers | 0.28 | **0.265** |
| Bayelsa | 0.20 | **0.202** |
| Edo | 0.07 | **0.082** |
| Imo | 0.05 | **0.051** |
| Akwa Ibom | 0.12 | **0.032** |
| Ondo | 0.03 | **0.005** |

Akwa Ibom was over-weighted roughly fourfold and Delta under-weighted by a third.
642 site-years fall outside every state polygon (offshore) and are now dropped
explicitly rather than being silently folded into onshore states.

## 3. Coverage claims do not match the data that exists

* The README and proposal claim **2012–2023, 84 state-years**. Site-level VIIRS
  volumes are published only for **2012–2020**; the 2021–2024 numbers in the notebook
  come from a hand-typed list of national GGFR totals, which cannot support a
  state-level analysis. The pipeline now keeps flare variables `NaN` outside the
  measured window instead of imputing, and the primary model window is **2012–2019**
  (the label needs year *t+1*) — 56 state-years, of which 21 end up in held-out folds.
* Because that is a very thin evidence base, a second specification with
  **conflict-history features only, 2000–2024** is run alongside it. ACLED covers
  Nigeria back to 1997, so this costs nothing and gives 119 held-out observations.
* The notebook loads ACLED from hard-coded dated filenames that are not in the repo,
  so it cannot be re-run by a reader. `src/silo/acquire.py` now resolves the current
  HDX resource URLs through the package API and downloads them.
* The PRIMAP workbook loaded in Phase 1 is never used in any model. It should be
  dropped or its role stated.

## 4. Baselines, benchmark and calibration (the requested additions)

All seven rows below come from the *same* rolling-origin folds: for each held-out
year, models see only strictly earlier years, and the label threshold is refitted on
those earlier years alone.

Conflict + satellite flaring, 2012–2019 (n = 21 held-out, 10 positive):

| model | precision | recall | F1 | ROC-AUC | PR-AUC | Brier |
|---|---|---|---|---|---|---|
| Neural Net (MLP) | 1.000 | 0.100 | 0.182 | 0.664 | 0.664 | 0.402 |
| Logistic Regression | 0.500 | 0.100 | 0.167 | 0.509 | 0.592 | 0.355 |
| SVM (RBF) | 0.000 | 0.000 | 0.000 | 0.564 | 0.576 | 0.348 |
| Random Forest | 1.000 | 0.100 | 0.182 | 0.536 | 0.576 | 0.317 |
| *Naive persistence* | 0.400 | 0.200 | 0.267 | 0.464 | 0.461 | 0.524 |
| XGBoost | 0.000 | 0.000 | 0.000 | 0.418 | 0.444 | 0.408 |
| *Historical base rate* | 0.167 | 0.100 | 0.125 | 0.386 | 0.419 | 0.407 |

Conflict history only, 2000–2024 (n = 119 held-out, 58 positive):

| model | precision | recall | F1 | ROC-AUC | PR-AUC | Brier |
|---|---|---|---|---|---|---|
| Neural Net (MLP) | 0.619 | 0.448 | 0.520 | 0.667 | 0.645 | 0.347 |
| SVM (RBF) | 0.606 | 0.345 | 0.440 | 0.590 | 0.569 | 0.274 |
| Random Forest | 0.545 | 0.414 | 0.471 | 0.560 | 0.563 | 0.281 |
| XGBoost | 0.500 | 0.362 | 0.420 | 0.561 | 0.554 | 0.324 |
| *Naive persistence* | 0.586 | 0.586 | **0.586** | 0.596 | 0.545 | 0.403 |
| Logistic Regression | 0.525 | 0.724 | 0.609 | 0.529 | 0.526 | 0.334 |
| *Historical base rate* | 0.167 | 0.017 | 0.031 | 0.401 | 0.419 | 0.345 |

What this says, honestly:

* **No model clearly beats "it happened last year".** The best AUCs are ~0.65–0.67,
  and naive persistence matches or beats several learners on F1. On the flare
  specification the ordering is barely distinguishable from noise at n = 21.
* **XGBoost is not the best model here.** Presenting it as the headline model needs
  either a justification other than performance (interpretability via SHAP is a
  legitimate one) or a change of model.
* **Probabilities are poorly calibrated** (Brier 0.27–0.52 against a base rate near
  0.5; see `outputs/figures/calibration_*.png`). A risk score used to prioritise
  community engagement must be calibrated or it should be presented as a *ranking*,
  not a probability.

This is a publishable result — "satellite flaring adds little predictive signal over
conflict history at state-year resolution, and simple baselines are competitive" is
a real finding — but it is a very different claim from the one the README currently
makes.

## 5. Label design

`slo_erosion = (future > p60) & (future > 1.2 × trailing 3-year mean)` is reasonable,
but three things need stating in the write-up:

* It is a **conflict-escalation proxy**, not a measure of social licence. The proposal
  is careful about this; the README is not ("early-warning system for SLO erosion").
* Prevalence is ~23% over 2000–2024 but ~49% inside the later held-out folds, because
  conflict is trending up and the threshold is refitted per fold. "Erosion" is
  therefore not a rare event in the evaluation period — worth reporting, as it makes
  precision/recall look better than they would for a rare-event task.
* The 1.2 multiplier and the 60th percentile are unmotivated. A sensitivity analysis
  over, say, {1.1, 1.2, 1.5} × {p50, p60, p75} would cost nothing and pre-empt the
  obvious reviewer question.

## 6. Other code issues

* **Double counting risk in ACLED.** Civilian-targeting events are a *subset* of
  political-violence events. The pipeline keeps them as an intensity feature
  (`civilian_share`) and never adds them to the total.
* **Leakage in features.** `state_history_mean` is an expanding mean of *past* years
  only (shifted); rolling windows all end at *t*. A regression test perturbs a future
  year and asserts earlier rows' features are unchanged.
* **No tests, no requirements file, no runnable entry point.** Added: `tests/`
  (9 tests), `requirements.txt`, `python -m silo.pipeline`.
* **Repository layout.** Nine PNGs and the risk map sit in the repo root while the
  README documents `models/`, `outputs/`, `data/`. New artefacts are written to the
  documented locations; the legacy root-level files should be moved or deleted.
* **Unpinned Colab-only workflow.** The notebook `pip install`s at run time and reads
  files from a manual upload. Nothing in it is reproducible off that machine.

## 7. Proposal-specific notes

* **Novelty claims.** "No system existed to connect those dots" / "Never done" are too
  strong. Satellite-flaring-to-conflict work exists (e.g. work linking gas flaring and
  oil infrastructure to local violence in the Niger Delta), and ACLED-based
  conflict forecasting is a large literature. Reframe the contribution as the
  *combination* (site-level flaring + explainability + an SLO framing), and support it
  with a documented search rather than an assertion.
* **Say what the label is.** Add one sentence early: "SLO erosion is operationalised
  as a conflict-escalation proxy; no direct measure of community consent exists at
  this resolution."
* **Reporting bias.** ACLED event counts reflect *reported* events. Media coverage in
  Rivers and Delta is denser than in Ondo, so state-level differences partly measure
  observation intensity. This deserves a paragraph, not a clause.
* **Sample size.** State a power caveat explicitly: 7 states × 8 usable years is not
  enough to separate five algorithms. If the intent is a serious model, move to
  LGA-level (ACLED has `Admin2`) or monthly resolution — both are available in the
  same files and would raise n by one to two orders of magnitude. **This is the single
  highest-value change available.**
* **Ethics section is good and should stay.** The misuse concern (a risk score used to
  pre-empt legitimate organising) is real; keeping the output at state level and
  framing it as an accountability tool is the right call.

## 8. Recommended next steps, in order

1. Move to **LGA-month** resolution (`Admin2` + month columns are already in the ACLED
   downloads). Everything else is downstream of having enough data.
2. Report the rolling-origin numbers above, with baselines, as the headline results.
3. Add threshold sensitivity analysis for the label.
4. Calibrate the chosen model (`CalibratedClassifierCV`, isotonic or Platt, fitted
   inside each fold) or present output as deciles rather than probabilities.
5. Rewrite README performance claims against §4, and move the legacy figures out of
   the repo root.
