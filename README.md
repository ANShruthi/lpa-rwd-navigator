# Lp(a) RWD Navigator — V1

Evidence-backed prototype for selecting fit-for-purpose real-world and observational data sources for lipoprotein(a) research.

## Included
- 10-source landscape dataset
- field-level evidence register
- transparent 0–3 scoring rubric
- Streamlit Dataset Explorer
- deterministic Study Matcher
- Data Gap Map
- Methodology/Evidence view

## Run
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Controlled values
Feature availability uses Yes / No / Partial / Limited / Unknown.
Unknown means not verified, not absent.

Scores are analyst judgments:
3 strong; 2 usable/moderate; 1 limited; 0 not suitable.

## Publication caveat
This is a sourced V1 prototype, not yet a finalized systematic-review dataset. Before manuscript submission, every material factual cell should receive field-level evidence and independent QC.
