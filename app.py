import streamlit as st
import pandas as pd
import numpy as np
import joblib
from io import BytesIO
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

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
    return psi * 0.0689655

def f_to_c(f):
    return (f - 32) * 5.0 / 9.0

# === UI Setup ===
st.set_page_config(page_title="Gas Wells Production Rate Predictor", layout="wide")
col1, col2, col3 = st.columns([1, 2, 1])
with col1:
    st.image("OIP.jfif", width=120)
with col2:
    st.markdown("<h1 style='font-size: 35px;'>Gas Wells Production Rate Predictor</h1>", unsafe_allow_html=True)
with col3:
    st.image("picocheiron_logo.jpg", width=120)

st.markdown("Upload a file or manually input well data to predict **Gas**, **Condensate**, and **Water** rates.")
option = st.radio("Choose input method:", ("Manual Input", "Upload Excel File"))

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
            thp_val = st.text_input("THP")
            choke_val = st.text_input("Choke (%)")
            flp_unit = st.selectbox("FLP Unit", ["bar", "psi"])
            flp_val = st.text_input("FLP")
        with col2:
            flt_unit = st.selectbox("Temperature Unit", ["°C", "°F"])
            flt_val = st.text_input("FLT")
            api_val = st.text_input("Oil Gravity (API)", value="44.1")
            gsg_val = st.text_input("Gas Specific Gravity", value="0.76")
        with col3:
            dp1_val = st.text_input("Venturi ΔP1 (mbar)")
            dp2_val = st.text_input("Venturi ΔP2 (mbar)")
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

                if thp_unit == "psi": thp = psi_to_bar(thp)
                if flp_unit == "psi": flp = psi_to_bar(flp)
                if flt_unit == "°F": flt = f_to_c(flt)

                row = pd.DataFrame([{ 'THP (bar)': thp, 'FLP (bar)': flp, 'Choke (%)': choke,
                                      'FLT ©': flt, 'Gas Specific Gravity': gsg, 'Oil Gravity (API)': api,
                                      'Venturi ΔP1 (mbar)': dp1, 'Venturi ΔP2 (mbar)': dp2 }])
                feat = engineer_features(row)
                X = pd.concat([row, feat.drop(columns=row.columns)], axis=1)
                X = X[expected_features]

                gas = np.clip(model_g.predict(X), 0, None)[0]
                cond = np.clip(model_c.predict(X), 0, None)[0]
                water = np.clip(model_w.predict(X), 0, None)[0]

                st.success("✅ Predicted Rates")
                st.markdown(f"""
                <div style='font-size: 20px; background-color: #111827; padding: 15px; border-radius: 10px;'>
                    🔷 <strong>Gas Rate:</strong> {gas:.2f} MMSCFD<br>
                    🛢️ <strong>Condensate Rate:</strong> {int(cond)} BPD<br>
                    💧 <strong>Water Rate:</strong> {int(water)} BPD
                </div>
                """, unsafe_allow_html=True)
        except ValueError:
            st.error("❗ Please enter only numeric values.")

# === Feedback Section ===
st.markdown("---")
st.subheader("📋 Feedback or Correction")
feedback = st.text_area("If you notice incorrect predictions or have additional notes, please share them below:")
excel_file = st.file_uploader("(Optional) Upload related Excel file", type=["xlsx"])

if st.button("Submit Feedback"):
    if feedback.strip():
        try:
            # Save feedback locally
            with open("user_feedback.txt", "a") as f:
                f.write(feedback + "\n\n")

            # Save feedback to Google Sheets
            creds_dict = st.secrets["gcp_service_account"]
            scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
            creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
            client = gspread.authorize(creds)
            sheet = client.open("Gas_Wells_Feedback").sheet1
            sheet.append_row([feedback])

            st.success("✅ Feedback successfully saved to Google Sheets.")

            # Save Excel to Drive and allow local download
            if excel_file:
                local_name = "uploaded_feedback_g.xlsx"
                with open(local_name, "wb") as f:
                    f.write(excel_file.read())

                with open(local_name, "rb") as f:
                    st.download_button("Download Uploaded Feedback Excel", data=f.read(), file_name=local_name)

        except Exception as e:
            st.error(f"❌ Failed to save feedback: {e}")
    else:
        st.warning("⚠️ Please enter some feedback before submitting.")
