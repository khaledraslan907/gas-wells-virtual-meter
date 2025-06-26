import streamlit as st
import pandas as pd
import numpy as np
import joblib
from io import BytesIO
import os

# === Load Models ===
base_path = os.path.dirname(os.path.abspath(__file__))

model_g = joblib.load(os.path.join(base_path, 'model_g_xgb.pkl'))
model_c = joblib.load(os.path.join(base_path, 'model_c_xgb.pkl'))
model_w = joblib.load(os.path.join(base_path, 'model_w_xgb.pkl'))

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
col1, col2 = st.columns([1, 1])
with col1:
    st.image("OIP.jfif", width=150)
with col2:
    st.image("picocheiron_logo.jpeg", width=150)

st.title("Gas Wells Production Rate Predictor")
st.markdown("Upload a file or manually input well data to predict **Gas**, **Condensate**, and **Water** rates.")

# === Input Method ===
option = st.radio("Choose input method:", ("Manual Input", "Upload Excel File"))

if option == "Manual Input":
    with st.form("manual_form"):
        thp = st.number_input('THP (bar)')
        flp = st.number_input('FLP (bar)')
        choke = st.number_input('Choke (%)')
        flt = st.number_input('FLT ©')
        gsg = st.number_input('Gas Specific Gravity')
        api = st.number_input('Oil Gravity (API)')
        dp1 = st.number_input('Venturi ΔP1 (mbar)')
        dp2 = st.number_input('Venturi ΔP2 (mbar)')
        submitted = st.form_submit_button("Predict")

    if submitted:
        row = pd.DataFrame([{
            'THP (bar)': thp, 'FLP (bar)': flp, 'Choke (%)': choke,
            'FLT ©': flt, 'Gas Specific Gravity': gsg, 'Oil Gravity (API)': api,
            'Venturi ΔP1 (mbar)': dp1, 'Venturi ΔP2 (mbar)': dp2
        }])
        feat = engineer_features(row)
        X = pd.concat([row, feat.drop(columns=row.columns)], axis=1)
        gas = np.clip(model_g.predict(X), 0, None)[0]
        cond = np.clip(model_c.predict(X), 0, None)[0]
        water = np.clip(model_w.predict(X), 0, None)[0]

        st.success("Predicted Rates")
        st.write(f"**Gas Rate:** {gas:.4f} MMSCFD")
        st.write(f"**Condensate Rate:** {cond:.4f} BPD")
        st.write(f"**Water Rate:** {water:.4f} BPD")

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
                st.error(f"Missing columns: {missing}. Please check your file.")
            else:
                feat_df = engineer_features(df_input)
                X_all = pd.concat([df_input, feat_df.drop(columns=df_input.columns)], axis=1)

                gas_pred = np.clip(model_g.predict(X_all), 0, None)
                cond_pred = np.clip(model_c.predict(X_all), 0, None)
                water_pred = np.clip(model_w.predict(X_all), 0, None)

                df_input['Gas Rate (MMSCFD)'] = gas_pred
                df_input['Condensate Rate (BPD)'] = cond_pred
                df_input['Water Rate (BPD)'] = water_pred

                st.success("✅ Predictions completed. Download your results below.")
                output = BytesIO()
                df_input.to_excel(output, index=False)
                st.download_button("Download Predictions", output.getvalue(), file_name="predicted_output.xlsx")
        except Exception as e:
            st.error(f"Something went wrong: {e}")
