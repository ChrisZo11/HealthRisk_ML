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

# Kita ganti st.sidebar.form menjadi st.form saja
with st.form("health_form"):
    
    # Bagi layar jadi 2 kolom agar inputan rapi (tidak memanjang ke bawah)
    col_input1, col_input2 = st.columns(2)
    
    with col_input1:
        st.markdown("**📂 Data Fisik & Biometrik**")
        age = st.slider("Usia (Tahun)", 0, 100, 30)
        bmi = st.number_input("BMI (Body Mass Index)", 10.0, 50.0, 22.0, step=0.1)
        sleep = st.slider("Durasi Tidur (Jam/hari)", 0.0, 12.0, 7.0, step=0.5)

    with col_input2:
        st.markdown("**🍷 Gaya Hidup**")
        # Pilihan Ya/Tidak
        smoking_opt = st.selectbox("Merokok?", ["Tidak (0)", "Ya (1)"])
        alcohol_opt = st.selectbox("Konsumsi Alkohol?", ["Tidak (0)", "Ya (1)"])
        sugar = st.slider("Asupan Gula (g/hari)", 0.0, 100.0, 30.0)

    st.markdown("---")
    # Tombol submit ada di tengah bawah form
    submitted = st.form_submit_button("🔍 Prediksi Sekarang", use_container_width=True)

# --- 5. Logika Prediksi ---
if submitted:
    # Konversi pilihan teks kembali ke angka 0 atau 1
    smoking_val = 1 if "Ya" in smoking_opt else 0
    alcohol_val = 1 if "Ya" in alcohol_opt else 0

    # Susun data (Urutan: Age, BMI, Smoking, Alcohol, Sleep, Sugar)
    input_data = np.array([[age, bmi, smoking_val, alcohol_val, sleep, sugar]])

    try:
        # Prediksi Decision Tree
        dt_pred = dt_model.predict(input_data)[0]

        # Prediksi KNN (Pakai Scaler)
        input_scaled = scaler.transform(input_data)
        knn_pred = knn_model.predict(input_scaled)[0]

        # Mapping Label
        label_map = {0: "High Risk (Beresiko Tinggi)", 1: "Low Risk (Aman)"}

        # --- 6. Tampilkan Hasil ---
        st.markdown("### 📊 Hasil Analisis")
        
        # Container hasil
        col_res1, col_res2 = st.columns(2)

        with col_res1:
            st.info("🌲 Model: **Decision Tree**")
            if dt_pred == 0:
                st.error(f"⚠️ {label_map[0]}")
            else:
                st.success(f"✅ {label_map[1]}")

        with col_res2:
            st.info("🔗 Model: **K-Nearest Neighbors**")
            if knn_pred == 0:
                st.error(f"⚠️ {label_map[0]}")
            else:
                st.success(f"✅ {label_map[1]}")

    except Exception as e:
        st.error(f"Terjadi kesalahan teknis: {e}")
