import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from io import BytesIO

# === Load Models ===
@st.cache_resource
def load_models():
    model_g = joblib.load("model_g_xgb.pkl")
    model_c = joblib.load("model_c_xgb.pkl")
    model_w = joblib.load("model_w_xgb.pkl")
    return model_g, model_c, model_w

model_g, model_c, model_w = load_models()

# === Feature Engineering ===
def engineer_features(df):
    df = df.copy()
    df['dP_ratio'] = df['Venturi ΔP2 (mbar)'] / df['Venturi ΔP1 (mbar)'].replace(0, 1e-3)
    df['flow_energy'] = df['THP (bar)'] * df['Venturi ΔP1 (mbar)']
    df['temp_pressure'] = df['FLT ©'] * df['THP (bar)']
    df['Pressure_ratio'] = df['FLP (bar)'] / df['THP (bar)'].replace(0, 1e-3)
    df['dp1_dp2'] = df['Venturi ΔP1 (mbar)'] * df['Venturi ΔP2 (mbar)']
    df['condensate_correction'] = df['temp_pressure'] / df['FLP (bar)'].replace(0, 1e-3)
    return df

# === UI Setup ===
st.set_page_config(page_title="Gas Wells Production Rate Predictor", layout="wide")

header_col1, header_col2, header_col3 = st.columns([1, 3, 1])
with header_col1:
    st.image("OIP.jfif", width=100)
with header_col2:
    st.markdown("""
    <div style='text-align: center; margin-top: 20px;'>
        <h1>Gas Wells Production Rate Predictor</h1>
    </div>
    """, unsafe_allow_html=True)
with header_col3:
    st.image("picocheiron_logo.jpg", width=100)

st.markdown("Upload a file or manually input well data to predict **Gas**, **Condensate**, and **Water** rates.")

# === Input Method ===
option = st.radio("Choose input method:", ("Manual Input", "Upload Excel File"))

# === Get feature names from model ===
def get_expected_features(model):
    if hasattr(model, "get_booster"):
        return model.get_booster().feature_names
    elif hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)
    else:
        return []

expected_features = get_expected_features(model_g)

# === Manual Input ===
if option == "Manual Input":
    with st.form("manual_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            thp = st.number_input('THP (bar)', help="Tubing Head Pressure", step=None)
            choke = st.number_input('Choke (%)', help="Choke Valve Opening (%)", step=None)
            flp = st.number_input('FLP (bar)', help="Flowline Pressure", step=None)
        with col2:
            flt = st.number_input('FLT ©', help="Flowline Temperature (°C)", step=None)
            api = st.number_input('Oil Gravity (API)', value=44.1, help="Oil Specific Gravity", step=None)
            gsg = st.number_input('Gas Specific Gravity', value=0.760, help="Gas Specific Gravity", step=None)
        with col3:
            dp1 = st.number_input('Venturi ΔP1 (mbar)', help="Venturi Differential Pressure 1", step=None)
            dp2 = st.number_input('Venturi ΔP2 (mbar)', help="Venturi Differential Pressure 2", step=None)
        submitted = st.form_submit_button("Predict")

    if submitted:
        try:
            row = pd.DataFrame([{
                'THP (bar)': thp, 'FLP (bar)': flp, 'Choke (%)': choke,
                'FLT ©': flt, 'Gas Specific Gravity': gsg, 'Oil Gravity (API)': api,
                'Venturi ΔP1 (mbar)': dp1, 'Venturi ΔP2 (mbar)': dp2
            }])
            feat = engineer_features(row)
            X = pd.concat([row, feat.drop(columns=row.columns)], axis=1)
            X = X[expected_features]

            gas = np.clip(model_g.predict(X), 0, None)[0]
            cond = np.clip(model_c.predict(X), 0, None)[0]
            water = np.clip(model_w.predict(X), 0, None)[0]

            st.success("✅ Predicted Rates")
            st.markdown(f"**Gas Rate:** {gas:.2f} MMSCFD")
            st.markdown(f"**Condensate Rate:** {int(cond)} BPD")
            st.markdown(f"**Water Rate:** {int(water)} BPD")

        except Exception as e:
            st.error(f"Prediction failed: {e}")

# === File Upload ===
else:
    uploaded_file = st.file_uploader("Upload Excel File", type=["xlsx"])

    if uploaded_file:
        try:
            df_input = pd.read_excel(uploaded_file)
            required_cols = ['THP (bar)', 'FLP (bar)', 'Choke (%)', 'FLT ©',
                             'Gas Specific Gravity', 'Oil Gravity (API)',
                             'Venturi ΔP1 (mbar)', 'Venturi ΔP2 (mbar)']
            missing = [col for col in required_cols if col not in df_input.columns]

            if missing:
                st.error(f"❌ Missing columns: {missing}. Please check your file.")
            else:
                df_input = df_input[required_cols]
                feat_df = engineer_features(df_input)
                X_all = pd.concat([df_input, feat_df.drop(columns=df_input.columns)], axis=1)
                X_all = X_all[expected_features]

                gas_pred = np.clip(model_g.predict(X_all), 0, None)
                cond_pred = np.clip(model_c.predict(X_all), 0, None)
                water_pred = np.clip(model_w.predict(X_all), 0, None)

                df_input['Gas Rate (MMSCFD)'] = gas_pred.round(2)
                df_input['Condensate Rate (BPD)'] = cond_pred.astype(int)
                df_input['Water Rate (BPD)'] = water_pred.astype(int)

                st.success("✅ Predictions completed. Download your results below.")
                output = BytesIO()
                df_input.to_excel(output, index=False, engine='openpyxl')
                st.download_button("Download Predictions", output.getvalue(), file_name="predicted_output.xlsx")
        except Exception as e:
            st.error(f"❌ Something went wrong: {e}")



