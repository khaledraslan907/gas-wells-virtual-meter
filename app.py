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

# === Unit Conversion Functions ===
def psi_to_bar(psi):
    return psi * 0.0689476

def f_to_c(f):
    return (f - 32) * 5.0 / 9.0

# === UI Setup ===
st.set_page_config(page_title="Gas Wells Production Rate Predictor", layout="wide")

# Layout
col1, col2, col3 = st.columns([1, 2, 1])

with col1:
    st.image("OIP.jfif", width=120)

with col2:
    st.markdown(
        """
        <div style="display: flex; align-items: flex-start; justify-content: flex-start; height: 100%;">
            <h1 style="margin-top: 10px; font-size: 35px;">Gas Wells Production Rate Predictor</h1>
        </div>
        """,
        unsafe_allow_html=True
    )

with col3:
    st.image("picocheiron_logo.jpg", width=120)

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
            thp_unit = st.selectbox("THP Unit", ["bar", "psi"])
            thp_val = st.text_input("THP", help="Tubing Head Pressure")
            choke_val = st.text_input("Choke (%)", help="Choke Valve Opening (%) (0-100)")
            flp_unit = st.selectbox("FLP Unit", ["bar", "psi"])
            flp_val = st.text_input("FLP", help="Flowline Pressure")

        with col2:
            flt_unit = st.selectbox("Temperature Unit", ["°C", "°F"])
            flt_val = st.text_input("FLT", help="Flowline Temperature")
            api_val = st.text_input("Oil Gravity (API)", value="44.1", help="Default: typical oil gravity")
            gsg_val = st.text_input("Gas Specific Gravity", value="0.76", help="Default: typical gas gravity")

        with col3:
            dp1_val = st.text_input("Venturi ΔP1 (mbar)", help="Venturi Differential Pressure 1")
            dp2_val = st.text_input("Venturi ΔP2 (mbar)", help="Venturi Differential Pressure 2")

        submitted = st.form_submit_button("Predict")

    if submitted:
        try:
            inputs = [thp_val, choke_val, flp_val, flt_val, api_val, gsg_val, dp1_val, dp2_val]
            if any(val.strip() == "" or float(val) < 0 for val in inputs):
                st.error("❗ Please enter only positive numeric values.")
            elif float(choke_val) > 100:
                st.error("❗ Choke percentage must be between 0 and 100.")
            else:
                thp = float(thp_val)
                flp = float(flp_val)
                flt = float(flt_val)
                choke = float(choke_val) / 100.0
                api = float(api_val)
                gsg = float(gsg_val)
                dp1 = float(dp1_val)
                dp2 = float(dp2_val)

                if thp_unit == "psi":
                    thp = psi_to_bar(thp)
                if flp_unit == "psi":
                    flp = psi_to_bar(flp)
                if flt_unit == "°F":
                    flt = f_to_c(flt)

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
                st.markdown(
                    f"""
                    <div style='font-size: 20px; padding: 15px; border-radius: 10px; background-color: #111827;'>
                        <p style='color:#00d2ff; margin-bottom:10px;'>
                            🔷 <strong>Gas Rate:</strong> {gas:.2f} MMSCFD
                        </p>
                        <p style='color:#2ca02c; margin-bottom:10px;'>
                            🛢️ <strong>Condensate Rate:</strong> {int(cond)} BPD
                        </p>
                        <p style='color:#d62728; margin-bottom:0;'>
                            💧 <strong>Water Rate:</strong> {int(water)} BPD
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        except ValueError:
            st.error("❗ Please enter only numeric values in all fields.")

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

# === Feedback Form ===
st.markdown("---")
st.markdown("### Feedback or Correction")
feedback = st.text_area("If you notice incorrect predictions or have additional notes, please share them below:")
if st.button("Submit Feedback"):
    if feedback.strip():
        with open("user_feedback.txt", "a") as f:
            f.write(feedback + "\n\n")
        st.success("✅ Thank you for your feedback!")
    else:
        st.warning("⚠️ Please enter some feedback before submitting.")
