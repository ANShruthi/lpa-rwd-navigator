Lp(a) RWD Navigator

An evidence-backed, fit-for-purpose landscape and decision-support prototype for identifying real-world and observational data sources relevant to lipoprotein(a) research.

What it does

The Navigator helps researchers compare data sources for Lp(a)-related questions across testing patterns, epidemiology and risk, cardiovascular outcomes, treatment patterns, health equity, HEOR/costs/utilization, and genetics.

Features

Dataset Explorer — searchable profiles of Lp(a)-relevant RWD and observational sources.

Study Matcher — deterministic fit-for-purpose scoring against predefined research use cases.

AI Study Assistant — translates a plain-English RWE question into structured study requirements; deterministic rules, not the language model, rank sources.

Potential alternatives & trade-offs — surfaces near-miss sources and explicitly identifies unmet hard requirements.

Gap Map — highlights data availability and documentation gaps.

Methodology & Evidence — scoring rubric and evidence provenance.

Decision architecture

Study question → structured requirements → evidence table → deterministic filtering/scoring → ranked sources + trade-offs

The AI layer is intentionally constrained to requirement extraction. It does not invent source characteristics or independently determine which database is “best.”

Fit-score interpretation

Score

Interpretation

3

Strong fit

2

Moderate fit

1

Limited fit

0

Not suitable / insufficient evidence

Unknown is not equivalent to No.

Current evidence base

The prototype currently characterizes 10 sources, including UK Biobank, All of Us, MESA, ARIC, CARDIA, Framingham Offspring, Jackson Heart Study, Epic Cosmos, TriNetX, and Optum Clinformatics with laboratory data.

Intended use

This project is designed for early RWE feasibility assessment, evidence planning, data-source landscape work, and portfolio demonstration.

Important limitation

Fit scores are evidence-informed feasibility assessments. They are not validated regulatory qualification, vendor endorsement, or a substitute for protocol-specific data feasibility, contracting/access review, data-quality assessment, or regulatory consultation. Publicly documented capabilities may differ from data actually available to a specific research team or license.

Technology

Python

Streamlit

pandas

OpenAI Responses API

CSV-based evidence and scoring layer

Repository structure

app.py
lpa_rwd_sources.csv
evidence.csv
scoring_rubric.csv
requirements.txt
README.md
LICENSE

Security

The OpenAI API key is stored through Streamlit Secrets and is not committed to this public repository.

Status

Portfolio/research prototype. Source characteristics and evidence should be periodically re-verified as datasets, documentation, and Lp(a) research evolve.
