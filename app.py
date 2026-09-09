
import streamlit as st
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# CSV files are stored in the repository root.
df = pd.read_csv(ROOT / "lpa_rwd_sources.csv").fillna("")
ev = pd.read_csv(ROOT / "evidence.csv").fillna("")
rubric = pd.read_csv(ROOT / "scoring_rubric.csv").fillna("")

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

    ranked = df[df["lpa_available"] == "Yes"].copy()

    if require_genetics:
        ranked = ranked[ranked["genetic_data"].isin(["Yes", "Partial"])]

    if require_costs:
        ranked = ranked[ranked["cost_data"].isin(["Yes", "Partial"])]

    if require_payer:
        ranked = ranked[ranked["insurance_payer"].isin(["Yes", "Partial"])]

    score_cols = [use_map[p] for p in priorities]

    if score_cols:
        ranked["fit_score"] = (
            ranked[score_cols]
            .apply(pd.to_numeric, errors="coerce")
            .fillna(0)
            .mean(axis=1)
        )
    else:
        ranked["fit_score"] = 0

    ranked = ranked.sort_values("fit_score", ascending=False)

    st.dataframe(
        ranked[
            [
                "source_name",
                "fit_score",
                "key_strengths",
                "key_limitations",
            ]
        ],
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
                    "require_lpa": {"type": "boolean"},
                    "summary": {"type": "string"}
                },
                "required": ["geography","priorities","require_genetics","require_costs","require_payer","require_lpa","summary"],
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
            ai_ranked = df.copy()
            if req["require_lpa"]:
                ai_ranked = ai_ranked[ai_ranked["lpa_available"] == "Yes"]
            if req["require_genetics"]:
                ai_ranked = ai_ranked[ai_ranked["genetic_data"].isin(["Yes","Partial"])]
            if req["require_costs"]:
                ai_ranked = ai_ranked[ai_ranked["cost_data"].isin(["Yes","Partial"])]
            if req["require_payer"]:
                ai_ranked = ai_ranked[ai_ranked["insurance_payer"].isin(["Yes","Partial"])]

            score_cols_ai = [use_map_ai[p] for p in req["priorities"] if p in use_map_ai]
            if score_cols_ai:
                ai_ranked["fit_score"] = ai_ranked[score_cols_ai].apply(
                    pd.to_numeric, errors="coerce"
                ).fillna(0).mean(axis=1)
            else:
                ai_ranked["fit_score"] = 0

            ai_ranked = ai_ranked.sort_values("fit_score", ascending=False)
            st.markdown("#### Fit-for-purpose ranking")
            st.dataframe(
                ai_ranked[["source_name","fit_score","key_strengths","key_limitations"]]
                .rename(columns={
                    "source_name":"Data source",
                    "fit_score":"Fit score (0–3)",
                    "key_strengths":"Why it may fit",
                    "key_limitations":"Important limitations"
                }),
                use_container_width=True,
                hide_index=True
            )
            st.caption("AI interprets the question; rankings come from the predefined dataset and scoring rubric.")
        except Exception as e:
            st.error("The AI assistant could not run. Check the OpenAI API key and package version in Streamlit Secrets/logs.")
            st.code(str(e))

with methods:
    st.subheader("Methodology")

    st.write(
        "Documented source characteristics are separated from analyst-assigned "
        "fit-for-purpose scores. Unknown means not verified, not absent."
    )

    st.dataframe(
        rubric,
        use_container_width=True,
        hide_index=True
    )

    st.subheader("Evidence register")

    st.dataframe(
        ev,
        use_container_width=True,
        hide_index=True
    )
