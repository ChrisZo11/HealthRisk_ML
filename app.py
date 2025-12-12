import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --- 1. Konfigurasi Halaman (Judul & Ikon) ---
st.set_page_config(
    page_title="Health Risk Check",
    page_icon="🏥",
    
)

# --- 2. Load Model & Scaler (Hanya sekali load biar cepat) ---
@st.cache_resource
def load_models():
    try:
        # Pastikan nama file ini sesuai dengan yang ada di folder kamu
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

# --- 3. Tampilan Utama ---
st.title("🏥 Health Risk Classification")
st.markdown("Analisis risiko kesehatan menggunakan **Decision Tree** & **KNN**.")
st.markdown("---")

# --- 4. Sidebar (Menu Kiri) ---
st.sidebar.header("📝 Masukkan Data")

with st.sidebar.form("health_form"):
    st.write("**Data Diri**")
    age = st.slider("Usia (Age)", 0, 100, 30)
    bmi = st.number_input("BMI", 10.0, 50.0, 22.0, step=0.1)
    
    st.write("**Gaya Hidup**")
    # Mengubah input 0/1 menjadi Pilihan Ya/Tidak agar lebih user friendly
    smoking_opt = st.selectbox("Merokok (Smoking)?", ["Tidak (0)", "Ya (1)"])
    alcohol_opt = st.selectbox("Alkohol (Alcohol)?", ["Tidak (0)", "Ya (1)"])
    
    sleep = st.slider("Tidur (Jam)", 0.0, 12.0, 7.0, step=0.5)
    sugar = st.slider("Gula (Sugar Intake)", 0.0, 100.0, 30.0)
    
    st.markdown("---")
    submitted = st.form_submit_button("🔍 Prediksi Sekarang")

# --- 5. Logika Prediksi ---
if submitted:
    # Konversi pilihan teks kembali ke angka 0 atau 1 untuk model
    smoking_val = 1 if "Ya" in smoking_opt else 0
    alcohol_val = 1 if "Ya" in alcohol_opt else 0

    # Susun data sesuai urutan waktu training (PENTING!)
    # Urutan: Age, BMI, Smoking, Alcohol, Sleep, Sugar
    input_data = np.array([[age, bmi, smoking_val, alcohol_val, sleep, sugar]])

    try:
        # 1. Prediksi Decision Tree
        dt_pred = dt_model.predict(input_data)[0]

        # 2. Prediksi KNN (Wajib pakai Scaler)
        input_scaled = scaler.transform(input_data)
        knn_pred = knn_model.predict(input_scaled)[0]

        # Mapping Label (0: High Risk, 1: Low Risk)
        label_map = {0: "High Risk (Beresiko Tinggi)", 1: "Low Risk (Aman)"}

        # --- 6. Tampilkan Hasil ---
        st.subheader("📊 Hasil Analisis")
        
        col1, col2 = st.columns(2)

        # Hasil Decision Tree
        with col1:
            st.info("🌲 Model: Decision Tree")
            if dt_pred == 0:
                st.error(f"⚠️ {label_map[0]}")
            else:
                st.success(f"✅ {label_map[1]}")

        # Hasil KNN
        with col2:
            st.info("🔗 Model: K-Nearest Neighbors")
            if knn_pred == 0:
                st.error(f"⚠️ {label_map[0]}")
            else:
                st.success(f"✅ {label_map[1]}")

    except Exception as e:
        st.error(f"Terjadi kesalahan saat prediksi: {e}")
else:
    # Pesan awal jika tombol belum ditekan
    st.info("👈 Silakan isi data di menu sebelah kiri, lalu klik tombol **Prediksi Sekarang**.")
