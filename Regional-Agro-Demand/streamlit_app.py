import streamlit as st
import sys
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ml.predict import predict_from_values, get_feature_names

st.set_page_config(page_title='Regional Agro Demand UI', layout='centered')

st.title('🌾 Regional Agro Demand Prediction')
st.write('Enter feature values to predict agro-demand cluster')

feature_names = get_feature_names()
values = []

for f in feature_names:
    values.append(st.number_input(f, value=0.0))

if st.button('Predict'):
    result = predict_from_values(values)
    st.success('Prediction Completed')
    st.json(result)
