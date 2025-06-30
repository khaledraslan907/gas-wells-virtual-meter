import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from io import BytesIO
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
import requests
# === Load Models ===
@st.cache_resource
def load_models():
    def download_model(url, filename):
        if not os.path.exists(filename):
            r = requests.get(url)
            with open(filename, 'wb') as f:
                f.write(r.content)
        return joblib.load(filename)

    model_g = download_model(st.secrets["model_g_url"], "model_g_xgb.pkl")
    model_c = download_model(st.secrets["model_c_url"], "model_c_xgb.pkl")
    model_w = download_model(st.secrets["model_w_url"], "model_w_xgb.pkl")

    return model_g, model_c, model_w

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
st.set_page_config(page_title="Gas Wells Virtual Meter", layout="wide")

# Layout
col1, col2, col3 = st.columns([1, 2, 1])

with col1:
    st.image("OIP.jfif", width=100)

with col2:
    st.markdown(
        """
        <div style="display: flex; align-items: flex-start; justify-content: flex-start; height: 100%;">
            <h1 style="margin-top: 10px; font-size: 35px;">Gas Wells Virtual Meter</h1>
        </div>
        """,
        unsafe_allow_html=True
    )

with col3:
    st.image("picocheiron_logo.jpg", width=100)

st.markdown("Upload a file or manually input well data to predict **Gas**, **Condensate**, and **Water** rates.")

# === Input Method ===
option = st.radio("Choose input method:", ("Manual Input", "Upload Excel File"))

# === Get feature names from model ===
model_g, model_c, model_w = load_models()

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
        colA, colB = st.columns(2)
        with colA:
            well_id = st.text_input("Well ID")
        with colB:
            date_val = st.date_input("Date")

        col1, col2, col3 = st.columns(3)
        with col1:
            thp_unit = st.selectbox("THP Unit", ["bar", "psi"])
            thp_val = st.text_input("THP")
            flp_unit = st.selectbox("FLP Unit", ["bar", "psi"])
            flp_val = st.text_input("FLP")
        with col2:
            flt_unit = st.selectbox("Temperature Unit", ["°C", "°F"])
            flt_val = st.text_input("FLT")
            api_val = st.text_input("Oil Gravity (API)", value="44.1")
            gsg_val = st.text_input("Gas Specific Gravity", value="0.76")
        with col3:
            choke_val = st.text_input("Choke (%)")
            dp1_val = st.text_input("Venturi ΔP1 (mbar)")
            dp2_val = st.text_input("Venturi ΔP2 (mbar)")

        submitted = st.form_submit_button("Predict")

    if submitted:
        session_df = st.session_state.get("prediction_table", pd.DataFrame())
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
                choke_percent = float(choke_val)
                choke = choke_percent / 100.0
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
                    "Well ID": well_id,
                    "Date": str(date_val),
                    'THP (bar)': thp, 'FLP (bar)': flp, 'Choke (%)': choke_percent,
                    'FLT ©': flt, 'Gas Specific Gravity': gsg, 'Oil Gravity (API)': api,
                    'Venturi ΔP1 (mbar)': dp1, 'Venturi ΔP2 (mbar)': dp2
                }])

                feat = engineer_features(row.drop(columns=["Well ID", "Date"]))
                X = pd.concat([
                    row.drop(columns=["Well ID", "Date", "Choke (%)"]),
                    pd.DataFrame({"Choke (%)": choke}, index=[0]),
                    feat.drop(columns=feat.columns.intersection(row.columns))
                ], axis=1)
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
                result_row = row.copy()
                result_row['Gas Rate (MMSCFD)'] = gas
                result_row['Condensate Rate (BPD)'] = int(cond)
                result_row['Water Rate (BPD)'] = int(water)

                session_df = pd.concat([session_df, result_row], ignore_index=True)
                
                st.session_state["prediction_table"] = session_df

                st.success("✅ Prediction completed.")
                st.dataframe(result_row)

        except ValueError:
            st.error("❗ Please enter only numeric values in all fields.")
# === Display Full Table and Clear Option ===
st.markdown("---")
st.subheader("📊 All Predictions This Session")

session_df = st.session_state.get("prediction_table", pd.DataFrame())

if not session_df.empty:
    st.dataframe(session_df)

    output = BytesIO()
    session_df.to_excel(output, index=False, engine='openpyxl')
    st.download_button("📥 Download All Predictions", output.getvalue(), file_name="well_predictions.xlsx")
else:
    st.info("No predictions yet.")

if st.button("🗑️ Clear Prediction Table"):
    st.session_state["prediction_table"] = pd.DataFrame()
    st.success("Prediction table cleared.")

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

# === Google Sheets Setup ===
def get_gsheet():
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    creds_dict = st.secrets["gcp_service_account"]
    creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
    client = gspread.authorize(creds)
    sheet = client.open("Gas-Wells-Virtual-Meter").sheet1
    return sheet

st.markdown("---")
st.markdown("### 📝 Feedback or Correction")

with st.form("feedback_form"):
    name = st.text_input("Your Name")
    well_id = st.text_input("Well ID")
    feedback_text = st.text_area("Your Feedback or Notes")
    feedback_file = st.file_uploader("Upload an Excel file (optional)", type=["xlsx"])
    submitted = st.form_submit_button("Submit Feedback")

if submitted:
    if not name or not well_id or not feedback_text.strip():
        st.warning("⚠️ Name, Well ID, and feedback are required.")
    else:
        try:
            # 1. Save to Google Sheets
            import gspread
            sheet_creds = service_account.Credentials.from_service_account_info(
                st.secrets["gcp_service_account"],
                scopes=["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
            )
            gc = gspread.authorize(sheet_creds)
            ws = gc.open("Gas-Wells-Virtual-Meter").sheet1

            timestamp = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
            ws.append_row([name, well_id, feedback_text.strip(), feedback_file.name if feedback_file else "None", timestamp])
            st.success("✅ Feedback saved to Google Sheets.")

            # 2. Upload Excel to Google Drive if provided
            FOLDER_ID = "1BvJMeCR2NpCxMls0wtyWFqJHmg6aZmJX"
            if feedback_file:
                filename = f"{well_id}_feedback_{timestamp.replace(':','-')}.xlsx"
            
                # Save temporarily
                with open(filename, "wb") as f:
                    f.write(feedback_file.read())
            
                # Upload to your shared folder in Google Drive
                drive_creds = sheet_creds.with_scopes(["https://www.googleapis.com/auth/drive"])
                drive_service = build("drive", "v3", credentials=drive_creds)
            
                file_metadata = {
                    "name": filename,
                    "parents": [FOLDER_ID]  # <-- Save in shared folder
                }
            
                media = MediaFileUpload(filename, mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
                uploaded = drive_service.files().create(
                    body=file_metadata,
                    media_body=media,
                    fields="id"
                ).execute()
            
                st.success("📁 Excel saved directly to your shared Drive folder.")


        except Exception as e:
            st.error(f"❌ Failed to save feedback: {repr(e)}")
