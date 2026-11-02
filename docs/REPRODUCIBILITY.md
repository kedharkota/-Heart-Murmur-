# Reproducibility and visual sources

## Scope

This guide describes what is available in the committed project and what is required to reproduce it. The documentation update inspected source and saved outputs; it did not rerun model training or hardware/simulation experiments.

## Required inputs

Obtain the CAMUS data through its official access process and reproduce the documented preprocessing and split. The app expects a YOLO model path, and experiment notebooks reference external training assets. The included plots and CSVs are historical experiment outputs. This is a research prototype, with no clinical validation established here.

## Entry point

Start with `PROJECT_DOCUMENTATION.md`. Extract `results.zip` to a separate local project directory, inspect its experiment notebooks, and restore their data and model paths.

The archive contains `app.py` and notebooks for experiments 2–5. Once the expected YOLO weights and Python dependencies are supplied, launch the archived demo from the extracted directory:

```bash
streamlit run app.py
```

The demo reads an optional `GEMINI_API_KEY` from the environment. A full environment lockfile and trained model weights are not included in the repository root.

## Evidence and interpretation

The committed `results/ef_metrics_summary.csv` inside the archive reports Exp 2 ejection-fraction MAE **4.8109 percentage points** and Pearson r **0.8621** on **100** cases. The archived Exp 2 segmentation summary reports mean DSC **0.9059**. These are source-reported experiment results, not independently reproduced measurements. The aggregate preview above comes from that same archive.

## Visual provenance

All screenshots and experiment figures are tied to repository material. Workflow SVGs are explanatory illustrations. The full image inventory is in [assets/sources.json](assets/sources.json).

| Visual | Source |
| --- | --- |
| [overview.svg](assets/overview.svg) | New explanatory project-workflow diagram based on the committed source and documentation; not a screenshot or measured result. |
| [evaluation.png](assets/evaluation.png) | results.zip/results/ef_scatter.png, original aggregate experiment output; not rerun |

## Validation

```bash
python tools/check_docs.py
```

The standard-library checker validates README/docs local link destinations and SVG/PNG/JPEG headers. It makes no external requests and installs no project runtime dependencies. External web links, model accuracy, image semantics, and full application execution need separate review.

## Reproduction record

For a new experiment or demo, record the source commit, environment versions, data provenance and split, configuration, command/notebook order, and resulting artifacts. Only label a measurement as reproduced after completing that run. Keep credentials and restricted datasets outside version control.
