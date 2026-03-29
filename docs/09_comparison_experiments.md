# Comparison Experiments (Step-by-Step)

This guide runs each comparison method one by one and saves outputs under `outputs/evaluation/`.

## 1) Prepare labels

Preferred: `data/labeled/labeled_events.csv` with `event_id,ground_truth_label`.  
Fallback: `data/labeled/auto_labeled_events.csv` (weak labels).

To generate an LLM-adjudicated silver-label set:

```bash
python scripts/generate_silver_labels.py --judge-model openai/gpt-4o-mini --judge-model google/gemini-2.0-flash-001
```

Recommended paper setting: use three judges and majority consensus:

```bash
python scripts/generate_silver_labels.py --judge-model openai/gpt-4o-mini --judge-model google/gemini-2.0-flash-001 --judge-model anthropic/claude-3.5-haiku --agreement-threshold 0.66 --confidence-threshold 0.6
```

## 2) Run one method at a time

Majority baseline:

```bash
python scripts/run_evaluation.py --baselines-only --methods majority_class
```

Rule baseline:

```bash
python scripts/run_evaluation.py --baselines-only --methods rule_based
```

Logistic Regression (OOF cross-validation):

```bash
python scripts/run_evaluation.py --baselines-only --methods logistic_regression
```

Random Forest (OOF cross-validation):

```bash
python scripts/run_evaluation.py --baselines-only --methods random_forest
```

Gradient Boosting (OOF cross-validation):

```bash
python scripts/run_evaluation.py --baselines-only --methods gradient_boosting
```

Agent predictions:

```bash
python scripts/run_evaluation.py --methods agent --predictions-file outputs/investigation_results.csv
```

Zero-shot LLM baseline (requires API key):

```bash
python scripts/run_evaluation.py --baselines-only --methods zero_shot_llm
```

## 3) Run all at once

```bash
python scripts/run_evaluation.py --methods all
```

## 4) Outputs

- Summary table: `outputs/evaluation/evaluation_summary.csv`
- Per-method predictions: `outputs/evaluation/<method>_predictions.csv`
- Silver-label summary: `outputs/labeling/llm_judge_summary.json`

## Notes

- ML baselines use out-of-fold prediction to reduce train/eval leakage.
- Matching is done by `event_id` (not row order).
- If your label file only has `event_id + label`, metadata is auto-filled from `outputs/anomaly_event_catalog.csv`.
- `zero_shot_llm` now reuses `outputs/evaluation/zero_shot_llm_predictions.csv` by default when it exists, so label changes can be rescored without another API call.
