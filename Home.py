import streamlit as st
import pandas as pd
import os

# Page config
st.set_page_config(page_title="Optimized Synthesis Gold Nanoparticles", layout="centered")

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
    background-color: #d2c8e0;
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
st.markdown('<h1 style="text-align:center;">LLM based optimized Synthesis of Gold Nanoparticles </h1>', unsafe_allow_html=True)

# Subtitle
st.markdown('<p style="text-align:center;">Predicting nanoparticle characteristics for cancer treatment using deep learning.\n Optimizing it using LLMs.</p>', unsafe_allow_html=True)

# Image
st.image("gold_nanoparticle.jpg", use_container_width=True)


# Model Description
st.markdown("""
###  About This Project:
This project integrates Deep Learning and a Large Language Model(LLM) to analyze and optimize gold nanoparticle synthesis for cancer treatment: it first loads and analyzes experimental data, including plotting correlations, then uses a trained Keras regression model and associated scalers to predict key nanoparticle properties based on user-provided synthesis parameters, and finally leverages a Groq API-accessed LLM to classify the suitability of the predicted properties against predefined criteria and offer targeted optimization suggestions for improving the synthesis method.

We've trained 4 **Deep Learning models** on experimental reaction data to predict:
- **Particle Size**
- **Zeta Potential**
- **Drug Loading Efficiency**
- **Targeting Efficiency**
- **Cytotoxicity**

Our Trained models include:
- **XGBoost**: A powerful tree-based model for regression tasks.
- **Keras**: A deep learning model for complex pattern recognition.
- **DCN**: Deep Cross Network for feature interaction learning.
- **MLP**: Multi-Layer Perceptron for non-linear regression.

Just enter your synthesis parameters, and our model will give optimized outcomes — no lab trials needed 
""")

# Call to Action
st.markdown('<h3 style="text-align:center;">Ready to explore your reaction outcomes?</h3>', unsafe_allow_html=True)


if st.button("Try the Model"):
    st.switch_page("pages/Model.py")