import streamlit as st
import pandas as pd

# Page config
st.set_page_config(page_title="Gold Nanocluster Project", layout="centered")

# Custom CSS styling
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600&display=swap');

    .main > div {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
    }

    div.stButton > button {
    background-color: #ffd700;
    color: black;
    font-weight: bold;
    padding: 0.75rem 1.5rem;
    border-radius: 12px;
    font-size: 1.1rem;
    margin-top: 25px;
    transition: all 0.3s ease;
    animation: fadeSlideIn 0.8s ease-out forwards;
    opacity: 0;
    transform: translateY(10px);
}

/* Hover glow + scale for all buttons */
button:hover {
    transform: scale(1.08);
    box-shadow: 0 8px 20px rgba(0, 0, 0, 0.2);
}
    @keyframes fadeSlideIn {
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    h1, h3, p {
        font-family: 'Poppins', sans-serif;
        animation: textFadeIn 1.5s ease-out forwards;
        opacity: 0;
        transform: translateY(20px);
    }

    @keyframes textFadeIn {
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    h1 { animation-delay: 0.2s; }
    h3 { animation-delay: 0.4s; }
    p { animation-delay: 0.6s; }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 style="text-align:center;">Gold Nanocluster Synth Predictor</h1>', unsafe_allow_html=True)

# Subtitle
st.markdown('<p style="text-align:center;">Predicting nanoparticle characteristics for cancer treatment using deep learning.</p>', unsafe_allow_html=True)

# Image
#st.image("https://upload.wikimedia.org/wikipedia/commons/6/63/Nanoparticles_targeting_cancer_cells.jpg", caption="Gold Nanoclusters targeting cancer cells", use_container_width=True)


# Model Description
st.markdown("""
###  About This Project:
Gold nanoclusters (AuNCs) are a new class of ultra-small nanoparticles showing huge promise in cancer therapy.  
Their size, charge, and surface properties can significantly influence **targeting efficiency**, **drug delivery**, and **cellular uptake**.

We've trained a **deep neural network (DNN)** on experimental reaction data to predict:
- **Particle Size**
- **Zeta Potential**
- **Polydispersity**
- **Drug Loading Efficiency**
- **Targeting Efficiency**
- **Cytotoxicity**

Just enter your synthesis parameters, and our model will give optimized outcomes — no lab trials needed 
""")

# Call to Action
st.markdown('<h3 style="text-align:center;">Ready to explore your reaction outcomes?</h3>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    if st.button("Go to Model A"):
        st.switch_page("Pages/Model A.py")

with col2:
    if st.button("Go to Model B"):
        st.switch_page("Pages/Model B.py")