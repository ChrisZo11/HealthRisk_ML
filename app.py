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

# --- 2. Load Model & Scaler (Cached) ---
@st.cache_resource
def load_models():
    try:
        dt_model = joblib.load("dt_best_model.sav")
        knn_model = joblib.load("knn_best_model.sav")
        scaler = joblib.load("scaler.sav")
        return dt_model, knn_model, scaler
    except FileNotFoundError as e:
        st.error(f"❌ File model tidak ditemukan: {e}. Pastikan file .sav ada di folder yang sama.")
        st.stop()
    except Exception as e:
        st.error(f"⚠️ Terjadi kesalahan saat memuat model: {e}")
        st.stop()

dt_model, knn_model, scaler = load_models()

# --- 3. UI: Judul & Deskripsi ---
st.title("🏥 Health Risk Classification System")
st.markdown("""
Aplikasi ini menggunakan **Decision Tree** dan **K-Nearest Neighbors (KNN)** untuk memprediksi risiko kesehatan berdasarkan gaya hidup dan data biometrik.
""")
st.markdown("---")

# --- 4. UI: Sidebar Input (Menggunakan Form) ---
st.sidebar.header("📝 Input Data Pasien")

with st.sidebar.form("health_form"):
    st.subheader("Data Pribadi")
    age = st.number_input("Usia (Tahun)", min_value=0, max_value=100, value=30, step=1)
    bmi = st.number_input("BMI (Body Mass Index)", min_value=10.0, max_value=60.0, value=22.0, step=0.1)
    
    st.subheader("Gaya Hidup")
    # Menggunakan label yang lebih manusiawi (Ya/Tidak) daripada 0/1
    smoking = st.radio("Merokok?", ["Tidak (0)", "Ya (1)"], index=0)
    alcohol = st.radio("Konsumsi Alkohol?", ["Tidak (0)", "Ya (1)"], index=0)
    
    col_sb1, col_sb2 = st.columns(2)
    with col_sb1:
        sleep = st.number_input("Tidur (Jam/hari)", min_value=0.0, max_value=24.0, value=7.0, step=0.5)
    with col_sb2:
        sugar = st.number_input("Gula (g/hari)", min_value=0.0, max_value=200.0, value=30.0, step=1.0)
        
    st.markdown("---")
    submitted = st.form_submit_button("🔍 Analisis Risiko")

# --- 5. Logika Prediksi ---
if submitted:
    # Konversi input radio button kembali ke integer (0 atau 1)
    smk_val = 1 if "Ya" in smoking else 0
    alc_val = 1 if "Ya" in alcohol else 0
    
    # 1. Siapkan Data (Urutan kolom HARUS sama dengan waktu training)
    # Kita pakai DataFrame biar lebih aman dan jelas nama kolomnya
    input_data = pd.DataFrame([[age, bmi, smk_val, alc_val, sleep, sugar]], 
                              columns=['Age', 'BMI', 'Smoking', 'Alcohol', 'Sleep', 'Sugar'])
    
    try:
        # 2. Scaling untuk KNN (PENTING: KNN butuh data yang discaling)
        input_scaled = scaler.transform(input_data)
        
        # 3. Prediksi
        dt_pred = dt_model.predict(input_data)[0]
        knn_pred = knn_model.predict(input_scaled)[0]
        
        # Mapping Label
        label_map = {0: "High Risk", 1: "Low Risk"}
        dt_label = label_map.get(dt_pred, "Unknown")
        knn_label = label_map.get(knn_pred, "Unknown")
        
        # --- 6. Tampilkan Hasil ---
        st.subheader("📊 Hasil Analisis")
        
        col1, col2 = st.columns(2)
        
        # Tampilan Decision Tree
        with col1:
            st.info("Model: Decision Tree")
            if dt_pred == 0: # High Risk
                st.error(f"⚠️ {dt_label}")
                st.markdown("Model Decision Tree mendeteksi pola berisiko tinggi.")
            else:
                st.success(f"✅ {dt_label}")
                st.markdown("Pola data terlihat aman menurut Decision Tree.")
                
        # Tampilan KNN
        with col2:
            st.info("Model: K-Nearest Neighbors (KNN)")
            if knn_pred == 0: # High Risk
                st.error(f"⚠️ {knn_label}")
                st.markdown("KNN (berdasarkan kemiripan tetangga) mendeteksi risiko tinggi.")
            else:
                st.success(f"✅ {knn_label}")
                st.markdown("Profil Anda mirip dengan kelompok risiko rendah.")

        # Disclaimer
        st.caption("---")
        st.warning("Catatan: Hasil ini adalah prediksi AI. Silakan konsultasi dengan dokter untuk diagnosis medis.")
        
    except Exception as e:
        st.error(f"Terjadi kesalahan saat prediksi: {e}")
        st.markdown("Tips: Pastikan urutan fitur (kolom) di scaler.sav sesuai dengan input.")

else:
    st.info("👈 Silakan isi data di sidebar kiri dan klik 'Analisis Risiko' untuk memulai.")
