
import streamlit as st
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# CSV files are stored in the repository root.
df = pd.read_csv(ROOT / "lpa_rwd_sources.csv").fillna("")
ev = pd.read_csv(ROOT / "evidence.csv").fillna("")
rubric = pd.read_csv(ROOT / "scoring_rubric.csv").fillna("")
design = pd.read_csv(ROOT / "v5_design_attributes.csv").fillna("")
weights_df = pd.read_csv(ROOT / "v5_tiebreaker_weights.csv").fillna(0)
df = df.merge(design, on=["source_id", "source_name"], how="left")

# ---- V4 fit-for-purpose matching helpers ---------------------------------
# Hard requirements are evaluated before use-case scoring.
# Yes = fully satisfies; Partial = eligible with a modest penalty;
# Limited = near-miss only; Unknown/No = does not satisfy the requirement.
CAPABILITY_PENALTY = {
    "Yes": 0.0,
    "Partial": 0.25,
    "Limited": 0.75,
    "Unknown": 1.0,
    "No": 1.0,
}

HARD_REQUIREMENT_FIELDS = {
    "genetics": "genetic_data",
    "cost data": "cost_data",
    "payer information": "insurance_payer",
    "pharmacy/dispensing data": "dispensing_claims",
    "longitudinal data": "longitudinal",
    "Lp(a) data": "lpa_available",
}


def _status(value):
    value = str(value).strip()
    return value if value in CAPABILITY_PENALTY else "Unknown"


def _geography_matches(country, geography):
    """Conservative geography matching for explicit study geographies."""
    g = str(geography or "").strip().lower()
    c = str(country or "").strip().lower()
    if not g or g in {"any", "global", "not specified", "unspecified"}:
        return True
    aliases = {
        "united states": {"united states", "us", "u.s.", "usa", "u.s.a."},
        "united kingdom": {"united kingdom", "uk", "u.k."},
        "canada": {"canada"},
        "denmark": {"denmark"},
        "iceland": {"iceland"},
        "europe": {"europe"},
    }
    for canonical, vals in aliases.items():
        if g in vals:
            if canonical == "europe":
                return c in {"europe", "denmark", "united kingdom"}
            return c == canonical
    return g in c or c in g


def evaluate_hard_requirements(row, requirements, geography=""):
    """Return eligibility tier, penalty, caveats, and failed requirements."""
    partials = []
    limited = []
    failed = []
    penalty = 0.0

    if geography and not _geography_matches(row.get("country", ""), geography):
        failed.append(f"geography ({geography})")

    for label, required in requirements.items():
        if not required:
            continue
        field = HARD_REQUIREMENT_FIELDS[label]
        value = _status(row.get(field, "Unknown"))
        if value == "Yes":
            continue
        if value == "Partial":
            partials.append(label)
            penalty += CAPABILITY_PENALTY[value]
        elif value == "Limited":
            limited.append(label)
            penalty += CAPABILITY_PENALTY[value]
        else:
            failed.append(f"{label} ({value})")
            penalty += CAPABILITY_PENALTY[value]

    # Only Yes/Partial can enter the primary candidate pool.
    # Limited, Unknown, No, or geography mismatch are surfaced as alternatives.
    if failed or limited:
        tier = "Alternative"
    elif partials:
        tier = "Conditional fit"
    else:
        tier = "Strong fit"

    caveats = []
    if partials:
        caveats.append("Partial: " + ", ".join(partials))
    if limited:
        caveats.append("Limited: " + ", ".join(limited))
    if failed:
        caveats.append("Unmet: " + ", ".join(failed))

    return tier, penalty, "; ".join(caveats), ", ".join(failed + limited)


def add_fit_scores(frame, score_cols, requirements, geography=""):
    out = frame.copy()
    if score_cols:
        out["base_fit_score"] = (
            out[score_cols].apply(pd.to_numeric, errors="coerce").fillna(0).mean(axis=1)
        )
    else:
        out["base_fit_score"] = 0.0

    evaluated = out.apply(
        lambda row: evaluate_hard_requirements(row, requirements, geography), axis=1
    )
    out[["fit_tier", "hard_req_penalty", "requirement_caveats", "unmet_hard_requirements"]] = pd.DataFrame(
        evaluated.tolist(), index=out.index
    )
    out["fit_score"] = (out["base_fit_score"] - out["hard_req_penalty"]).clip(lower=0)
    tier_order = {"Strong fit": 0, "Conditional fit": 1, "Alternative": 2}
    out["_tier_order"] = out["fit_tier"].map(tier_order).fillna(3)
    return out.sort_values(["_tier_order", "fit_score", "base_fit_score"], ascending=[True, False, False])


def fit_label(score):
    score = float(score)
    if score >= 2.5:
        return f"Strong — {score:.1f}"
    if score >= 1.5:
        return f"Moderate — {score:.1f}"
    if score > 0:
        return f"Limited — {score:.1f}"
    return "Not suitable — 0.0"


V5_ATTRIBUTE_FIELDS = [
    "routine_care_score", "scale_score", "followup_score", "outcome_quality_score",
    "assay_score", "genetics_depth_score", "medication_depth_score",
    "economic_data_score", "diversity_sdoh_score", "repeated_lpa_score",
]

def _priority_weight_map(priority_names):
    """Average the predefined design-quality weights across selected priorities."""
    if not priority_names:
        return {f: 0.0 for f in V5_ATTRIBUTE_FIELDS}
    rows = weights_df[weights_df["priority"].isin(priority_names)]
    if rows.empty:
        return {f: 0.0 for f in V5_ATTRIBUTE_FIELDS}
    return {f: float(pd.to_numeric(rows[f], errors="coerce").fillna(0).mean()) for f in V5_ATTRIBUTE_FIELDS}

def add_v5_design_score(frame, priority_names):
    out = frame.copy()
    wmap = _priority_weight_map(priority_names)
    total_weight = sum(wmap.values())
    if total_weight <= 0:
        out["design_quality_score"] = 0.0
        return out
    score = 0
    for field, weight in wmap.items():
        score = score + pd.to_numeric(out[field], errors="coerce").fillna(0) * weight
    out["design_quality_score"] = score / total_weight
    return out

def v5_rank(frame, priority_names):
    """Hard requirements and use-case fit remain primary; design quality breaks ties."""
    out = add_v5_design_score(frame, priority_names)
    return out.sort_values(
        ["_tier_order", "fit_score", "design_quality_score", "base_fit_score"],
        ascending=[True, False, False, False],
    )

def recommendation_tier(row, max_primary_design=None):
    if row["fit_tier"] == "Conditional fit":
        return "Conditional fit"
    if row["fit_tier"] != "Strong fit":
        return "Alternative"
    if float(row["fit_score"]) >= 2.5:
        if max_primary_design is None or float(row["design_quality_score"]) >= max_primary_design - 0.35:
            return "Primary recommendation"
        return "Other strong fit"
    return "Other strong fit"

st.set_page_config(
    page_title="Lp(a) RWD Navigator",
    page_icon="🧭",
    layout="wide"
)

st.title("Lp(a) RWD Navigator")
st.caption(
    "Evidence-backed, fit-for-purpose landscape of real-world and observational "
    "data sources for lipoprotein(a) research."
)

explorer, matcher, gaps, methods = st.tabs(
    ["Dataset Explorer", "Study Matcher", "Gap Map", "Methodology & Evidence"]
)

with explorer:
    st.subheader("Explore data sources")

    q = st.text_input(
        "Search sources or capabilities",
        placeholder="e.g., genetics, claims, testing, UK Biobank"
    )

    view = df.copy()

    if q:
        mask = (
            view.astype(str)
            .apply(lambda col: col.str.contains(q, case=False, na=False))
            .any(axis=1)
        )
        view = view[mask]

    display_cols = [
        "source_name",
        "data_type",
        "country",
        "approx_population",
        "lpa_available",
        "cv_outcomes_use",
        "health_equity_use",
        "heor_use",
        "genetics_use",
        "verification_status",
    ]

    pretty_view = view[display_cols].rename(columns={
        "source_name": "Data source",
        "data_type": "Data type",
        "country": "Geography",
        "approx_population": "Approx. population",
        "lpa_available": "Lp(a) available",
        "cv_outcomes_use": "CV outcomes fit",
        "health_equity_use": "Health equity fit",
        "heor_use": "HEOR fit",
        "genetics_use": "Genetics fit",
        "verification_status": "Evidence status",
    })
    st.dataframe(
        pretty_view,
        use_container_width=True,
        hide_index=True
    )

    if len(view):
        name = st.selectbox(
            "Open source profile",
            view["source_name"].tolist()
        )

        r = view[view["source_name"] == name].iloc[0]

        st.markdown(f"### {name}")

        c1, c2, c3 = st.columns(3)
        c1.metric("CV outcomes fit", f'{r["cv_outcomes_use"]}/3')
        c2.metric("HEOR fit", f'{r["heor_use"]}/3')
        c3.metric("Genetics fit", f'{r["genetics_use"]}/3')

        st.write(r["population_description"])
        st.markdown("**Strengths:** " + str(r["key_strengths"]))
        st.markdown("**Limitations:** " + str(r["key_limitations"]))
        st.markdown(
            "**Published Lp(a) example:** " + str(r["published_lpa_example"])
        )

        # Support either exact source IDs or pooled evidence rows containing multiple IDs.
        source_id = str(r["source_id"])
        source_ev = ev[
            ev["source_id"]
            .astype(str)
            .apply(lambda x: source_id in x.split("|"))
        ]

        if len(source_ev):
            st.markdown("#### Supporting evidence")
            st.dataframe(
                source_ev[
                    [
                        "field_name",
                        "value",
                        "citation",
                        "publication_year",
                        "confidence",
                        "url",
                    ]
                ],
                use_container_width=True,
                hide_index=True,
            )

with matcher:
    st.subheader("Study Matcher")
    st.write(
        "Select the research priorities that matter most. "
        "The ranking is deterministic and based on the scoring rubric."
    )

    use_map = {
        "Testing patterns": "testing_patterns_use",
        "Epidemiology / risk": "epidemiology_use",
        "Cardiovascular outcomes": "cv_outcomes_use",
        "Treatment patterns": "treatment_patterns_use",
        "Health equity": "health_equity_use",
        "HEOR / costs / utilization": "heor_use",
        "Genetics": "genetics_use",
    }

    priorities = st.multiselect(
        "Research priorities",
        list(use_map.keys()),
        default=["Cardiovascular outcomes"],
    )

    require_genetics = st.checkbox("Require genetics/genomics")
    require_costs = st.checkbox("Require cost data")
    require_payer = st.checkbox("Require insurance/payer information")
    require_pharmacy = st.checkbox("Require pharmacy/dispensing data")
    require_longitudinal = st.checkbox("Require longitudinal data")

    manual_requirements = {
        "genetics": require_genetics,
        "cost data": require_costs,
        "payer information": require_payer,
        "pharmacy/dispensing data": require_pharmacy,
        "longitudinal data": require_longitudinal,
        "Lp(a) data": True,
    }

    score_cols = [use_map[p] for p in priorities]
    ranked = add_fit_scores(df, score_cols, manual_requirements)
    ranked = v5_rank(ranked, priorities)

    primary = ranked[ranked["fit_tier"].isin(["Strong fit", "Conditional fit"])].copy()
    alternatives = ranked[ranked["fit_tier"] == "Alternative"].copy()

    st.markdown("#### Eligible sources")
    if len(primary):
        primary["Fit"] = primary["fit_score"].map(fit_label)
        max_design = primary.loc[primary["fit_score"] >= 2.5, "design_quality_score"].max() if (primary["fit_score"] >= 2.5).any() else None
        primary["Recommendation"] = primary.apply(lambda r: recommendation_tier(r, max_design), axis=1)
        primary["Design tie-breaker"] = primary["design_quality_score"].map(lambda x: f"{float(x):.2f}/3")
        st.dataframe(
            primary[
                ["source_name", "Recommendation", "Fit", "Design tie-breaker", "requirement_caveats", "key_strengths", "key_limitations"]
            ].rename(columns={
                "source_name":"Data source",
                "requirement_caveats":"Requirement caveats",
                "key_strengths":"Why it may fit",
                "key_limitations":"Important limitations",
            }),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.warning("No source in the current evidence base satisfies all selected hard requirements with documented Yes/Partial capability.")

    if len(alternatives):
        st.markdown("#### Potential alternatives and trade-offs")
        alt = alternatives.head(8).copy()
        alt["Fit"] = alt["fit_score"].map(fit_label)
        st.dataframe(
            alt[["source_name", "Fit", "unmet_hard_requirements", "key_strengths", "key_limitations"]]
            .rename(columns={
                "source_name":"Data source",
                "unmet_hard_requirements":"Unmet / insufficiently documented hard requirements",
                "key_strengths":"Why it may still be useful",
                "key_limitations":"Important limitations",
            }),
            use_container_width=True,
            hide_index=True,
        )

with gaps:
    st.subheader("Data Gap Map")

    fields = [
        "lpa_available",
        "mortality",
        "dispensing_claims",
        "cost_data",
        "insurance_payer",
        "race_ethnicity",
        "socioeconomic_variables",
        "genetic_data",
        "imaging_data",
    ]

    rows = []

    for field in fields:
        counts = df[field].value_counts()
        rows.append(
            {
                "feature": field,
                "Yes": int(counts.get("Yes", 0)),
                "Partial": int(counts.get("Partial", 0)),
                "Limited": int(counts.get("Limited", 0)),
                "Unknown": int(counts.get("Unknown", 0)),
                "No": int(counts.get("No", 0)),
            }
        )

    gap = pd.DataFrame(rows)

    st.dataframe(
        gap,
        use_container_width=True,
        hide_index=True
    )

    st.bar_chart(
        gap.set_index("feature")[["Yes", "Partial"]]
    )


st.divider()
st.subheader("AI Study Assistant")
st.caption("Describe a study question in plain English. AI extracts study requirements; the app's deterministic rubric remains responsible for ranking data sources.")

study_question = st.text_area(
    "Describe your RWE study",
    placeholder="Example: I want to evaluate cardiovascular outcomes and healthcare utilization among U.S. patients with elevated Lp(a). I need longitudinal lab measurements, pharmacy data and costs."
)

if study_question:
    if "OPENAI_API_KEY" not in st.secrets:
        st.info("AI assistant is ready in the code. Add OPENAI_API_KEY in Streamlit Secrets to activate it.")
    else:
        try:
            from openai import OpenAI
            import json

            client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
            schema = {
                "type": "object",
                "properties": {
                    "geography": {"type": "string"},
                    "priorities": {
                        "type": "array",
                        "items": {"type": "string", "enum": [
                            "Testing patterns", "Epidemiology / risk",
                            "Cardiovascular outcomes", "Treatment patterns",
                            "Health equity", "HEOR / costs / utilization", "Genetics"
                        ]}
                    },
                    "require_genetics": {"type": "boolean"},
                    "require_costs": {"type": "boolean"},
                    "require_payer": {"type": "boolean"},
                    "require_pharmacy": {"type": "boolean"},
                    "require_longitudinal": {"type": "boolean"},
                    "require_lpa": {"type": "boolean"},
                    "summary": {"type": "string"}
                },
                "required": ["geography","priorities","require_genetics","require_costs","require_payer","require_pharmacy","require_longitudinal","require_lpa","summary"],
                "additionalProperties": False
            }

            response = client.responses.create(
                model="gpt-5.6",
                instructions=(
                    "You extract RWE study design requirements. Do not recommend a database. "
                    "Only translate the user's question into the supplied schema. "
                    "If a requirement is not stated, do not invent it."
                ),
                input=study_question,
                text={"format": {
                    "type": "json_schema",
                    "name": "study_requirements",
                    "strict": True,
                    "schema": schema
                }}
            )
            req = json.loads(response.output_text)
            st.markdown("#### Interpreted requirements")
            st.write(req["summary"])

            use_map_ai = {
                "Testing patterns":"testing_patterns_use",
                "Epidemiology / risk":"epidemiology_use",
                "Cardiovascular outcomes":"cv_outcomes_use",
                "Treatment patterns":"treatment_patterns_use",
                "Health equity":"health_equity_use",
                "HEOR / costs / utilization":"heor_use",
                "Genetics":"genetics_use",
            }
            ai_requirements = {
                "genetics": req["require_genetics"],
                "cost data": req["require_costs"],
                "payer information": req["require_payer"],
                "pharmacy/dispensing data": req["require_pharmacy"],
                "longitudinal data": req["require_longitudinal"],
                "Lp(a) data": req["require_lpa"],
            }

            score_cols_ai = [use_map_ai[p] for p in req["priorities"] if p in use_map_ai]
            ai_all = add_fit_scores(df, score_cols_ai, ai_requirements, req["geography"])
            ai_all = v5_rank(ai_all, req["priorities"])

            eligible = ai_all[ai_all["fit_tier"].isin(["Strong fit", "Conditional fit"])].copy()
            alternatives = ai_all[ai_all["fit_tier"] == "Alternative"].copy()

            # Keep the primary panel focused on credible candidates. Very low adjusted scores
            # are surfaced with alternatives even if all hard requirements are technically Partial.
            primary = eligible[eligible["fit_score"] >= 2.0].copy()
            low_scoring_eligible = eligible[eligible["fit_score"] < 2.0].copy()
            if len(low_scoring_eligible):
                low_scoring_eligible["fit_tier"] = "Alternative"
                low_scoring_eligible["unmet_hard_requirements"] = low_scoring_eligible["requirement_caveats"].replace("", "Low fit-for-purpose score")
                alternatives = pd.concat([alternatives, low_scoring_eligible], ignore_index=False)
                alternatives = alternatives.sort_values(["fit_score", "design_quality_score", "base_fit_score"], ascending=[False, False, False])

            st.markdown("#### Best-fit sources")
            if len(primary):
                primary["Fit"] = primary["fit_score"].map(fit_label)
                max_design = primary.loc[primary["fit_score"] >= 2.5, "design_quality_score"].max() if (primary["fit_score"] >= 2.5).any() else None
                primary["Recommendation"] = primary.apply(lambda r: recommendation_tier(r, max_design), axis=1)
                primary["Design tie-breaker"] = primary["design_quality_score"].map(lambda x: f"{float(x):.2f}/3")
                st.dataframe(
                    primary[["source_name","Recommendation","Fit","Design tie-breaker","requirement_caveats","key_strengths","key_limitations"]]
                    .rename(columns={
                        "source_name":"Data source",
                        "requirement_caveats":"Requirement caveats",
                        "key_strengths":"Why it may fit",
                        "key_limitations":"Important limitations"
                    }),
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.warning("No source in the current evidence base satisfies the stated hard requirements with sufficient documented fit.")

            if len(alternatives):
                st.markdown("#### Potential alternatives and trade-offs")
                near = alternatives.head(7).copy()
                near["Fit"] = near["fit_score"].map(fit_label)
                st.dataframe(
                    near[["source_name","Fit","unmet_hard_requirements","key_strengths","key_limitations"]]
                    .rename(columns={
                        "source_name":"Data source",
                        "unmet_hard_requirements":"Unmet / insufficiently documented hard requirements",
                        "key_strengths":"Why it may still be useful",
                        "key_limitations":"Important limitations"
                    }),
                    use_container_width=True,
                    hide_index=True
                )

            st.caption(
                "AI interprets the study question. Rankings and trade-offs are generated from the predefined "
                "evidence table and scoring rubric; the model does not invent source capabilities."
            )
        except Exception as e:
            msg = str(e).lower()
            if "insufficient_quota" in msg or "no credits remaining" in msg:
                st.warning("AI Study Assistant is temporarily unavailable because API usage credits are not available. The deterministic Dataset Explorer and Study Matcher remain fully functional.")
            elif "authentication" in msg or "api key" in msg:
                st.warning("AI Study Assistant is temporarily unavailable because its API connection could not be authenticated.")
            else:
                st.warning("AI Study Assistant is temporarily unavailable. The rest of the navigator remains fully functional.")

with methods:
    st.subheader("Methodology")
    st.markdown("### Interpreting fit scores")
    st.markdown("""
- **3 — Strong fit:** source has direct, well-documented capability for the use case.
- **2 — Moderate fit:** usable capability with meaningful limitations or incomplete coverage.
- **1 — Limited fit:** capability is indirect, restricted, or suitable only for selected analyses.
- **0 — Not suitable / insufficient evidence:** required capability is absent or not supported by the current evidence base.

`Unknown` is treated differently from `No`: lack of public documentation does not establish that a variable or capability is absent.

**V5 matching rule:** hard requirements are evaluated first. `Yes` fully satisfies a requirement; `Partial` remains eligible with an explicit penalty/caveat; `Limited`, `Unknown`, and `No` do not enter the primary best-fit pool. The 0–3 use-case score ranks eligible sources. When sources tie or nearly tie, an evidence-derived design-quality layer breaks ties using attributes such as routine-care representativeness, analytic scale, follow-up, outcome ascertainment, Lp(a) assay documentation, genetics depth, medication/dispensing depth, payer/cost depth, diversity/SDOH depth, and repeated Lp(a) potential. These tie-breaker attributes do not override hard requirements or the primary fit score.
""")
    st.caption("Source characteristics, fit assessments, and V5 design-quality attributes should be re-verified before protocol finalization, vendor contracting, regulatory submission, manuscript submission, or other consequential use.")

    st.write(
        "Documented source characteristics are separated from analyst-assigned "
        "fit-for-purpose scores. Unknown means not verified, not absent."
    )

    st.dataframe(
        rubric,
        use_container_width=True,
        hide_index=True
    )

    st.subheader("V5 design-quality tie-breaker")
    st.write(
        "The tie-breaker is used only after hard-requirement eligibility and use-case fit. "
        "Scores are working evidence-derived assessments on a 0–3 scale and should be re-verified before external use."
    )
    design_display = design.copy()
    st.dataframe(design_display, use_container_width=True, hide_index=True)

    st.subheader("Evidence register")

    st.dataframe(
        ev,
        use_container_width=True,
        hide_index=True
    )
