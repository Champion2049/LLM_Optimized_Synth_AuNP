import os
import sys

import streamlit as st

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import theme

st.set_page_config(
    page_title="Optimized Synthesis of Gold Nanoparticles",
    layout="centered",
)
theme.apply_base_style()

theme.page_header(
    "Optimized Synthesis of Gold Nanoparticles",
    subtitle=(
        "Predict nanoparticle properties for cancer-treatment applications with deep "
        "learning, then optimize the synthesis using a deterministic, "
        "rule engine."
    ),
    kicker="Computational Nanochemistry",
)

if os.path.exists("gold_nanoparticle.jpg"):
    st.image("gold_nanoparticle.jpg", use_container_width=True)

# --- Overview -----------------------------------------------------------------
theme.section_title("Overview")
st.markdown(
    "A trained regression model predicts the key properties of a gold-nanoparticle "
    "synthesis from its reaction parameters. Each predicted property is then scored against "
    "validated suitability ranges, and every property that falls outside its range is mapped "
    "to a **pre-validated optimization recommendation** — indexed by *(synthesis method, "
    "property, direction of deviation)* and accompanied by a one-line mechanistic "
    "justification that can be checked against the literature."
)

# --- What it covers -----------------------------------------------------------
col_props, col_models = st.columns(2)
with col_props:
    st.markdown('<div class="field-label">Properties predicted</div>', unsafe_allow_html=True)
    st.markdown(
        "- Particle size\n"
        "- Particle width\n"
        "- Drug-loading efficiency\n"
        "- Targeting efficiency\n"
        "- Cytotoxicity"
    )
with col_models:
    st.markdown('<div class="field-label">Models evaluated</div>', unsafe_allow_html=True)
    st.markdown(
        "- **Keras** — deep neural network for multi-output regression\n"
        "- **XGBoost** — gradient-boosted trees\n"
        "- **DCN** — deep & cross network for feature interactions\n"
        "- **MLP** — multi-layer perceptron baseline"
    )

st.markdown("")
st.markdown(
    "Enter a set of synthesis parameters and the tool returns the predicted outcome, the "
    "properties that are out of range, and the specific adjustments to bring them back — "
    "without a lab trial."
)

st.divider()
if st.button("Open the predictor", type="primary"):
    st.switch_page("pages/Model.py")
