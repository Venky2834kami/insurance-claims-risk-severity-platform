import streamlit as st, pandas as pd, joblib
from src.genai import deterministic_explanation
st.set_page_config(page_title='Claims Severity Platform', layout='wide'); st.title('Insurance Claims Risk & Severity Platform')
model_path=st.sidebar.text_input('Model path','artifacts/model.joblib')
file=st.file_uploader('Upload a CSV of new claims', type='csv')
if file and st.button('Score claims'):
    bundle=joblib.load(model_path); df=pd.read_csv(file); df['predicted_loss']=bundle['model'].predict(df).clip(0); st.dataframe(df); st.metric('Average predicted severity', f'{df.predicted_loss.mean():,.2f}'); st.info(deterministic_explanation(df.predicted_loss.mean()))
