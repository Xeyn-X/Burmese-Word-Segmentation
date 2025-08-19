import os
import streamlit as st
from streamlit_extras.stylable_container import stylable_container

from inference_crf import CRFInference
from inference_LSTM import LSTMInference
from inference_biLSTM import BiLSTMInference
from inference_biLSTMCRF import BiLSTMCRFInference
from inference_biLSTMAttention import BiLSTMAttentionInference
from inference_biLSTMAttention_CRF import BiLSTMAttentionCRFInference


# --- Instantiate Models ---
crf_predict = CRFInference(model_path="weight/crf_model.pkl")

lstm_predict = LSTMInference(
    config_path="config.json",
    vocab_path="vocab.json",
    model_path="weight/best_LSTM_model.pth"
)

bilstm_predict = BiLSTMInference(
    config_path="config.json",
    vocab_path="vocab.json",
    model_path="weight/best_BiLSTM_model.pth"
)

bilstm_crf_predict = BiLSTMCRFInference(
    config_path="config.json",
    vocab_path="vocab.json",
    model_path="weight/best_biLSTMCRF_model.pth"
)

bilstm_attention_predict = BiLSTMAttentionInference(
    config_path="config.json",
    vocab_path="vocab.json",
    model_path="weight/best_biLSTMAttention_model.pth"
)

bilstm_attention_crf_predict = BiLSTMAttentionCRFInference(
    config_path="config.json",
    vocab_path="vocab.json",
    model_path="weight/best_biLSTMAttentionCRF_model.pth"
)


# --- Page Settings ---
st.set_page_config(page_title="Word Segmentation", layout="centered")

# --- Hide Streamlit UI Elements ---
st.markdown("""
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# --- Load Custom CSS ---
css_path = "./style.css"
if os.path.exists(css_path):
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# --- Page Title ---
st.markdown("<h1 style='text-align: center;'>Word Segmentation</h1>", unsafe_allow_html=True)
st.markdown("<h6 style='text-align: center; color: #b8520a;'>Comparing Statistical, Neural and Hybrid Methods</h6>", unsafe_allow_html=True)


# --- Input Area ---
input_text = st.text_area("Input Sentence", placeholder="Enter Input ...", key="input")

# --- Stylable Button ---
with stylable_container(
    "btn",
    css_styles="""
        button {
            background-color: #b8520a;
            color: white;
            display: block;
            margin: 40px auto;
            padding: 10px 30px;
            border-radius: 25px;
            font-weight: 600;
            font-size: 16px;
            text-align: center;
        }
        button:hover p {
            color: #d7d9d8;
        }
        .st-emotion-cache-13lcgu3:active {
            background-color: #de6c1a;
            border: none;
        }
        button:active p {
            color: #d7d9d8;
        }
    """,
):
    run_segmentation = st.button("Segmentation")


# --- Segmentation and Display ---
if run_segmentation and input_text.strip():
    models = {
        "CRF": lambda: crf_predict.segment_text(input_text),
        "LSTM": lambda: lstm_predict.predict(input_text),
        "BiLSTM": lambda: bilstm_predict.predict(input_text),
        "BiLSTM-CRF": lambda: bilstm_crf_predict.predict(input_text),
        "BiLSTM-ATTENTION": lambda: bilstm_attention_predict.predict(input_text),
        "BiLSTM-ATTENTION-CRF": lambda: bilstm_attention_crf_predict.predict(input_text),
    }

    for model_name, func in models.items():
        try:
            prediction = func()
        except Exception as e:
            prediction = f"<Error: {e}>"

        with st.expander(model_name, expanded=True):
            st.markdown(
                f"<span style='color: #b8520a; font-size: 15px;'>{prediction}</span>",
                unsafe_allow_html=True
            )


# --- Footer ---
st.markdown(
    "<div class='custom-footer'>"
    "Copyright © 2025 Hein Htet Arkar Mg "
    "(Supported by National Artificial Intelligence Improvement Project-NAIIP)"
    "</div>",
    unsafe_allow_html=True
)