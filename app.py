import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --- 1. Konfigurasi Halaman ---
st.set_page_config(
    page_title="Health Risk Check",
    page_icon="🏥",
    layout="wide"
)

# --- 2. Load Model & Scaler ---
@st.cache_resource
def load_models():
    try:
        dt_model = joblib.load("dt_best_model.sav")
        knn_model = joblib.load("knn_best_model.sav")
        scaler = joblib.load("scaler.sav")
        return dt_model, knn_model, scaler
    except FileNotFoundError:
        st.error("❌ File model (.sav) tidak ditemukan. Pastikan file ada di folder yang sama.")
        st.stop()
    except Exception as e:
        st.error(f"⚠️ Error loading model: {e}")
        st.stop()

dt_model, knn_model, scaler = load_models()

# --- 3. Tampilan Judul ---
st.title("🏥 Health Risk Classification")
st.markdown("Analisis risiko kesehatan menggunakan **Decision Tree** & **KNN**.")
st.markdown("---")

# --- 4. FORM INPUT DI TENGAH (Main Page) ---
st.subheader("📝 Masukkan Data Pasien")

with st.form("health_form"):
    
    col_input1, col_input2 = st.columns(2)
    
    with col_input1:
        st.markdown("**📂 Data Fisik & Biometrik**")
        age = st.slider("Usia (Tahun)", 0, 100, 30)
        bmi = st.number_input("BMI (Body Mass Index)", 10.0, 50.0, 22.0, step=0.1)
        sleep = st.slider("Durasi Tidur (Jam/hari)", 0.0, 12.0, 7.0, step=0.5)

    with col_input2:
        st.markdown("**🍷 Gaya Hidup**")
        smoking_opt = st.selectbox("Merokok?", ["Tidak (0)", "Ya (1)"])
        alcohol_opt = st.selectbox("Konsumsi Alkohol?", ["Tidak (0)", "Ya (1)"])
        sugar = st.slider("Asupan Gula (g/hari)", 0.0, 100.0, 30.0)

    st.markdown("---")
    submitted = st.form_submit_button("🔍 Prediksi Sekarang", use_container_width=True)

# --- 5. Logika Prediksi ---
if submitted:
    
    smoking_val = 1 if "Ya" in smoking_opt else 0
    alcohol_val = 1 if "Ya" in alcohol_opt else 0

    input_data = np.array([[age, bmi, smoking_val, alcohol_val, sleep, sugar]])

    try:
        # --- PREDIKSI PROBABILITAS ---
        
        # 1. Decision Tree Probabilitas
        dt_proba = dt_model.predict_proba(input_data)[0]
        dt_pred = np.argmax(dt_proba) # Dapatkan kelas hasil prediksi (0 atau 1)

        # 2. KNN Probabilitas (Wajib pakai Scaler)
        input_scaled = scaler.transform(input_data)
        knn_proba = knn_model.predict_proba(input_scaled)[0]
        knn_pred = np.argmax(knn_proba) # Dapatkan kelas hasil prediksi (0 atau 1)

        # Mapping Label
        label_map = {0: "High Risk (Beresiko Tinggi)", 1: "Low Risk (Aman)"}

        # --- 6. Tampilkan Hasil ---
        st.markdown("### 📊 Hasil Analisis")
        
        col_res1, col_res2 = st.columns(2)

        # Hasil Decision Tree
        with col_res1:
            st.info("🌲 Model: **Decision Tree**")
            if dt_pred == 0:
                st.error(f"⚠️ {label_map[0]}")
            else:
                st.success(f"✅ {label_map[1]}")
            
            st.markdown(f"**Tingkat Keyakinan Model:**")
            col_dt1, col_dt2 = st.columns(2)
            col_dt1.metric("Probabilitas High Risk", f"{dt_proba[0]*100:.2f}%")
            col_dt2.metric("Probabilitas Low Risk", f"{dt_proba[1]*100:.2f}%")


        # Hasil KNN
        with col_res2:
            st.info("🔗 Model: **K-Nearest Neighbors**")
            if knn_pred == 0:
                st.error(f"⚠️ {label_map[0]}")
            else:
                st.success(f"✅ {label_map[1]}")

            st.markdown(f"**Tingkat Keyakinan Model:**")
            col_knn1, col_knn2 = st.columns(2)
            col_knn1.metric("Probabilitas High Risk", f"{knn_proba[0]*100:.2f}%")
            col_knn2.metric("Probabilitas Low Risk", f"{knn_proba[1]*100:.2f}%")

        st.caption("---")
        st.caption("Disclaimer: Probabilitas menunjukkan keyakinan model, bukan diagnosis medis.")

    except Exception as e:
        st.error(f"Terjadi kesalahan teknis saat prediksi: {e}")
        st.caption("Pastikan model yang digunakan mendukung fungsi .predict_proba()")
else:
    st.info("Masukkan data dan klik tombol prediksi.")
