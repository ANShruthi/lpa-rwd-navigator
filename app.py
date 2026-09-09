
import streamlit as st
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent
df = pd.read_csv(ROOT/"data"/"lpa_rwd_sources.csv").fillna("")
ev = pd.read_csv(ROOT/"data"/"evidence.csv").fillna("")
rubric = pd.read_csv(ROOT/"data"/"scoring_rubric.csv").fillna("")

st.set_page_config(page_title="Lp(a) RWD Navigator", page_icon="🧭", layout="wide")
st.title("Lp(a) RWD Navigator")
st.caption("Evidence-backed, fit-for-purpose landscape of data sources for lipoprotein(a) research.")

explorer, matcher, gaps, methods = st.tabs(["Dataset Explorer","Study Matcher","Gap Map","Methodology & Evidence"])

with explorer:
    q = st.text_input("Search sources or capabilities")
    view = df.copy()
    if q:
        mask = view.astype(str).apply(lambda col: col.str.contains(q, case=False, na=False)).any(axis=1)
        view = view[mask]
    cols = ["source_name","data_type","country","approx_population","lpa_available",
            "cv_outcomes_use","health_equity_use","heor_use","genetics_use","verification_status"]
    st.dataframe(view[cols], use_container_width=True, hide_index=True)
    if len(view):
        name = st.selectbox("Open source profile", view["source_name"].tolist())
        r = view[view["source_name"]==name].iloc[0]
        st.subheader(name)
        c1,c2,c3 = st.columns(3)
        c1.metric("CV outcomes fit", f'{r["cv_outcomes_use"]}/3')
        c2.metric("HEOR fit", f'{r["heor_use"]}/3')
        c3.metric("Genetics fit", f'{r["genetics_use"]}/3')
        st.write(r["population_description"])
        st.markdown("**Strengths:** " + str(r["key_strengths"]))
        st.markdown("**Limitations:** " + str(r["key_limitations"]))
        src_ev = ev[ev["source_id"].astype(str).str.contains(str(r["source_id"]), regex=False)]
        if len(src_ev):
            st.markdown("#### Supporting evidence")
            st.dataframe(src_ev[["field_name","value","citation","publication_year","confidence","url"]],
                         use_container_width=True, hide_index=True)

with matcher:
    st.subheader("Study Matcher")
    use_map = {
        "Testing patterns":"testing_patterns_use",
        "Epidemiology / risk":"epidemiology_use",
        "Cardiovascular outcomes":"cv_outcomes_use",
        "Treatment patterns":"treatment_patterns_use",
        "Health equity":"health_equity_use",
        "HEOR / costs / utilization":"heor_use",
        "Genetics":"genetics_use",
    }
    priorities = st.multiselect("Research priorities", list(use_map), default=["Cardiovascular outcomes"])
    require_genetics = st.checkbox("Require genetics/genomics")
    require_costs = st.checkbox("Require cost data")
    require_payer = st.checkbox("Require insurance/payer information")
    ranked = df[df["lpa_available"]=="Yes"].copy()
    if require_genetics:
        ranked = ranked[ranked["genetic_data"].isin(["Yes","Partial"])]
    if require_costs:
        ranked = ranked[ranked["cost_data"].isin(["Yes","Partial"])]
    if require_payer:
        ranked = ranked[ranked["insurance_payer"].isin(["Yes","Partial"])]
    sc = [use_map[p] for p in priorities]
    ranked["fit_score"] = ranked[sc].mean(axis=1) if sc else 0
    ranked = ranked.sort_values("fit_score", ascending=False)
    st.dataframe(ranked[["source_name","fit_score","key_strengths","key_limitations"]],
                 use_container_width=True, hide_index=True)

with gaps:
    st.subheader("Data Gap Map")
    fields = ["lpa_available","mortality","dispensing_claims","cost_data","insurance_payer",
              "race_ethnicity","socioeconomic_variables","genetic_data","imaging_data"]
    out = []
    for f in fields:
        vc = df[f].value_counts()
        out.append({"feature":f,"Yes":int(vc.get("Yes",0)),"Partial":int(vc.get("Partial",0)),
                    "Limited":int(vc.get("Limited",0)),"Unknown":int(vc.get("Unknown",0)),
                    "No":int(vc.get("No",0))})
    gap = pd.DataFrame(out)
    st.dataframe(gap, use_container_width=True, hide_index=True)
    st.bar_chart(gap.set_index("feature")[["Yes","Partial"]])

with methods:
    st.subheader("Methodology")
    st.write("Documented characteristics are separated from analyst-assigned fit-for-purpose scores. Unknown means not verified, not absent.")
    st.dataframe(rubric, use_container_width=True, hide_index=True)
    st.subheader("Evidence register")
    st.dataframe(ev, use_container_width=True, hide_index=True)
