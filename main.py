import os
import streamlit as st
from streamlit_extras.stylable_container import stylable_container
from lstm_predict import LSTMPredict
from bilstm_predict import BiLSTMPredict
from crf_predict import CRFPredict
from bilstm_crf_predict import BiLSTMCRFPredict

# --- Instantiate Models ---
lstm_predict = LSTMPredict(
    model_path="weight/word_seg_lstm.pth",
    config_path="data_config/data_config_lstm.pkl"
)

bilstm_predict = BiLSTMPredict(
    model_path="weight/word_seg_bilstm.pth",
    config_path="data_config/data_config_bilstm.pkl"
)

crf_predict = CRFPredict(
    model_path="weight/word_seg_crf.pkl"
)

bilstm_crf_predict = BiLSTMCRFPredict(
    model_path="weight/word_seg_bilstm_crf.pth",
    config_path="data_config/data_config_bilstm_crf.pkl"
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
css_path = './style.css'
if os.path.exists(css_path):
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# --- Page Title ---
st.markdown("<h1 style='text-align: center;'>Word Segmentation</h1>", unsafe_allow_html=True)
st.markdown("<h6 style='text-align: center; color: #b8520a;'>Using LSTM, BiLSTM, CRF and BiLSTM-CRF</h6>", unsafe_allow_html=True)

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
    button = st.button("Segmentation")

# --- Segmentation and Display ---
if button and input_text.strip():
    try:
        pred_lstm = lstm_predict.segment_text(input_text)
    except Exception as e:
        pred_lstm = f"<Error: {e}>"

    try:
        pred_bilstm = bilstm_predict.segment_text(input_text)
    except Exception as e:
        pred_bilstm = f"<Error: {e}>"

    try:
        pred_crf = crf_predict.segment_text(input_text)
    except Exception as e:
        pred_crf = f"<Error: {e}>"

    try:
        pred_bilstm_crf = bilstm_crf_predict.segment_text(input_text)
    except Exception as e:
        pred_bilstm_crf = f"<Error: {e}>"

    with st.expander("LSTM", expanded=True):
        st.markdown(f"<span style='color: #b8520a;font-size:15px;'>{pred_lstm}</span>", unsafe_allow_html=True)


    with st.expander("BiLSTM", expanded=True):
        st.markdown(f"<span style='color: #b8520a;font-size:15px;'>{pred_bilstm}</span>", unsafe_allow_html=True)

    with st.expander("CRF", expanded=True):
        st.markdown(f"<span style='color: #b8520a;font-size:15px;'>{pred_crf}</span>", unsafe_allow_html=True)

    with st.expander("BiLSTM-CRF", expanded=True):
        st.markdown(f"<span style='color: #b8520a;font-size:15px;'>{pred_bilstm_crf}</span>", unsafe_allow_html=True)


# --- Footer ---
st.markdown(
    "<div class='custom-footer'>Copyright © 2025 မြန်မာစာဖွံ့ဖြိုးတိုးတက်ရေးစီမံကိန်း (MLLIP)</div>",
    unsafe_allow_html=True
)
