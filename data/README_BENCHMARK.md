# Event-level reference benchmark

This directory contains the manuscript-ready event-level reference benchmark:

- `human_review_benchmark_199.csv`

The file contains 199 candidate wastewater SARS-CoV-2 anomaly events used for benchmark scoring in the manuscript. Each row corresponds to one merged anomaly event anchored to its event peak date.

## Field definitions

- `event_id`: Unique event identifier.
- `site_id`: Wastewater surveillance site identifier.
- `start_date`: Start date of the merged anomaly event, formatted as `YYYY-MM-DD`.
- `end_date`: End date of the merged anomaly event, formatted as `YYYY-MM-DD`.
- `peak_date`: Event anchor date, defined as the date with the largest absolute rolling z-score within the merged anomaly event, formatted as `YYYY-MM-DD`.
- `duration_days`: Duration of the merged event in days.
- `peak_zscore`: Rolling z-score at the event peak date.
- `mean_zscore`: Mean rolling z-score across the merged event window.
- `detection_methods`: Anomaly detectors contributing to the event call, when available.
- `detector_vote_count`: Maximum number of anomaly detectors supporting the event.
- `final_reference_label`: Operational reference label used for benchmark scoring.
- `agreement_tier`: Reviewer-agreement tier, reported as `full_agreement`, `majority_agreement`, or `no_majority`.
- `agreement_ratio`: Fractional agreement among the three independent reviews.
- `reviewer_1_label`, `reviewer_2_label`, `reviewer_3_label`: Labels assigned independently by the three reviewers.
- `reviewer_1_confidence`, `reviewer_2_confidence`, `reviewer_3_confidence`: Reviewer confidence scores on the original 0--1 scale.
- `final_label_rule`: Rule used to assign the operational reference label.
- `curation_note`: Short note describing how the operational reference label was assigned.

## Notes

The manuscript-ready benchmark file omits legacy exploratory silver-label fields and model-judge provenance fields. Those exploratory fields were not used as the public reference benchmark for the final manuscript.
