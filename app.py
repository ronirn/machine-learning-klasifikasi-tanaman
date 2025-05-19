import streamlit as st
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
from st_aggrid import AgGrid, GridOptionsBuilder
import plotly.figure_factory as ff
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
from PIL import Image
import base64
import random
from fpdf import FPDF
import io
from streamlit_option_menu import option_menu

# -----------------------------
# 🔧 Config halaman
st.set_page_config(page_title="PlantClassifier", layout="wide", page_icon="🌱")

# Memuat model dan scaler dan label encoder
with open('model/naive_bayes_model.pkl', 'rb') as f_nb:
    nb_model = pickle.load(f_nb)

with open('model/decision_tree_model.pkl', 'rb') as f_dt:
    dt_model = pickle.load(f_dt)

with open('model/scaler.pkl', 'rb') as f_scaler:
    scaler = pickle.load(f_scaler)

with open('model/label_encoder.pkl', 'rb') as f:
    le = pickle.load(f)

# -----------------------------
# 🎨 Custom CSS
st.markdown("""
    <style>
    .center-text { text-align: center; }
    .small-img img { width: 100px !important; height: auto; }
    .nav-button button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px;
        font-weight: bold;
        margin-bottom: 10px;
    }
    .galeri-img {
        border-radius: 12px;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.15);
        width: 100px;
        height: 100px;
        object-fit: contain;
        margin-bottom: 12px;
        background-color: #f9f9f9;
        padding: 6px;
    }
    .caption {
        text-align: center;
        font-size: 0.75rem;
        color: #333;
    }
    </style>
""", unsafe_allow_html=True)

# -----------------------------
# Sidebar dengan Logo di atas
with st.sidebar:
    # 🖼️ Tampilkan logo
    logo = Image.open("image/logo.png")  # Ganti dengan path logo kamu
    st.image(logo, width=250)  # Bisa atur lebar sesuai keinginan

    # 🔥 Menu navigasi
    selected = option_menu(
        "Menu Utama", 
        ["Beranda", "Klasifikasi", "Visualisasi", "Evaluasi", "Tentang"],
        icons=["house", "kanban", "bar-chart", "clipboard-data", "info-circle"],
        menu_icon="cast", 
        default_index=0,
        styles={
            "container": {"padding": "5px", "background-color": "#f0f2f6"},
            "icon": {"color": "green", "font-size": "20px"}, 
            "nav-link": {"font-size": "16px", "text-align": "left", "margin":"5px", "--hover-color": "#eee"},
            "nav-link-selected": {"background-color": "#4CAF50"},
        }
    )

# -----------------------------
# Helper Function to Convert Image to Base64
def get_base64_image(img_path):
    with open(img_path, "rb") as img_file:
        img_data = img_file.read()
        return base64.b64encode(img_data).decode()

# 🚀 Navigasi antar halaman
if selected == "Beranda":
    # -----------------------------
    # 🖼️ Banner atas
    banner_img = "image/banner.png"
    st.markdown(f"""
        <div style="border: 2px solid #4CAF50; border-radius: 12px; overflow: hidden; width: 100%; max-width: 1200px; margin: auto; box-shadow: 0 4px 20px rgba(0,0,0,0.1);">
            <img src="data:image/png;base64,{get_base64_image(banner_img)}" style="width: 100%; height: auto;">
        </div>
    """, unsafe_allow_html=True)

    # -----------------------------
    # 🌱 Konten Beranda
    st.title("Aplikasi Klasifikasi Tanaman")
    st.subheader("Berbasis Machine Learning - Naïve Bayes & Decision Tree")

    with st.expander(" **Apa itu Aplikasi Klasifikasi Tanaman?**"):
        st.markdown("""
        Aplikasi Klasifikasi Tanaman adalah sebuah sistem berbasis web yang menggunakan teknologi *Machine Learning* untuk memprediksi jenis tanaman yang paling sesuai dengan kondisi tanah dan iklim yang dimasukkan oleh pengguna.

        Tujuan utama aplikasi ini adalah membantu:
        -  **Petani**, dalam menentukan tanaman terbaik untuk ditanam berdasarkan lahan mereka.
        -  **Peneliti**, dalam mengakses data klasifikasi tanaman dengan cepat.
        -  **Pengambil keputusan**, dalam perencanaan pertanian dan ketahanan pangan berbasis data.

        Aplikasi ini menggunakan dua algoritma utama:
        - **Naïve Bayes**
        - **Decision Tree**

        yang terbukti efektif dalam klasifikasi berbasis data numerik dan kategorikal.

        """)

    with st.expander(" **Mengapa Menggunakan Aplikasi Ini?**"):
        st.markdown("""
        -  **Mudah digunakan**: Antarmuka sederhana dan intuitif, cocok untuk semua kalangan.
        -  **Cepat dan akurat**: Didukung oleh algoritma Machine Learning yang telah diuji.
        -  **Visualisasi hasil**: Tersedia grafik dan metrik evaluasi model.
        -  **Komprehensif**: Mendukung klasifikasi untuk lebih dari 20 jenis tanaman umum di Indonesia.
        -  **Fleksibel**: Bisa digunakan untuk perencanaan jangka pendek maupun jangka panjang.

        **Yuk mulai klasifikasi! Klik "Klasifikasi" di menu sidebar.**
        """)
        
    with st.expander(" **Parameter yang Digunakan**"):
        st.markdown("Berikut adalah parameter tanah dan iklim yang digunakan dalam klasifikasi tanaman beserta pengaruhnya:")
        data = {
            "Parameter": [
                "N (Nitrogen)", 
                "P (Phosphorus)", 
                "K (Potassium)", 
                "Temperature (°C)", 
                "Humidity (%)", 
                "pH", 
                "Rainfall (mm)"
            ],
            "Keterangan": [
                "Nutrisi utama untuk pertumbuhan daun dan batang tanaman.",
                "Mendukung pembentukan akar, bunga, dan buah.",
                "Meningkatkan kekuatan batang dan ketahanan terhadap penyakit.",
                "Menentukan kenyamanan iklim tumbuh tanaman, memengaruhi metabolisme.",
                "Kelembapan udara yang berpengaruh pada transpirasi dan fotosintesis.",
                "Menunjukkan keasaman/alkalinitas tanah yang mempengaruhi penyerapan nutrisi.",
                "Curah hujan penting untuk suplai air tanah dan pertumbuhan tanaman."
            ],
            "Tingkat Pengaruh": [
                "Sangat Penting",
                "Penting", 
                "Penting", 
                "Sangat Penting", 
                "Sedang", 
                "Penting", 
                "Sangat Penting"
            ]
        }
        df = pd.DataFrame(data)
        st.table(df)
    st.markdown("####  Jelajahi berbagai fitur di Menu Sidebar")    
    # -----------------------------
    # 🎯 Ajakan Jelajahi Sidebar
    st.info("🔍 **Jelajahi berbagai fitur di sidebar** untuk melakukan klasifikasi tanaman, melihat riwayat prediksi, mengunduh hasil, dan lainnya.")
    # 💡 Tips Pertanian
    st.markdown("---")
    st.markdown("###  Tips Pertanian Hari Ini")
    st.success(random.choice([
        " Gunakan pH tanah netral (6.0 - 7.5) untuk hasil pertanian optimal.",
        " Curah hujan yang baik meningkatkan pertumbuhan tanaman.",
        " Pemupukan seimbang (NPK) sangat penting.",
        " Data tanah & iklim bantu pilih tanaman terbaik."
    ]))

    # ----------------------------------------------------------
    # 🖼️ Galeri Tanaman
    st.markdown("---")
    st.markdown("### Contoh Tanaman yang didukung")

    tanaman = [
        ("image/padi.png", "Padi"),
        ("image/jagung.png", "Jagung"),
        ("image/pisang.png", "Pisang"),
        ("image/cabaimerah.png", "Cabai"),
        ("image/kelapa.png", "Kelapa"),
        ("image/mangga.png", "Mangga"),
        ("image/kakao.png", "Kakao"),
        ("image/semangka.png", "Semangka"),
        ("image/jeruk.png", "Jeruk"),
        ("image/sawi.png", "Sawi")
    ]

    for row in range(2):
        cols = st.columns(5)
        for i in range(5):
            idx = row * 5 + i
            if idx < len(tanaman):  # Adding check to avoid IndexError
                img_path, caption = tanaman[idx]
                with cols[i]:
                    st.markdown(f"""
                        <div style="text-align: center;">
                            <img src="data:image/png;base64,{get_base64_image(img_path)}" class="galeri-img"/>
                            <div class="caption">{caption}</div>
                        </div>
                    """, unsafe_allow_html=True)
    # -----------------------------

elif selected == "Klasifikasi":
    # -----------------------------
    # Halaman Klasifikasi
    st.title("Klasifikasi Tanaman")
    st.subheader("Input Data Tanah & Iklim")

    st.markdown("""
    Silakan masukkan parameter tanah dan iklim di bawah ini untuk memprediksi jenis tanaman yang cocok.
    """)

    with st.form(key='form_klasifikasi'):
        col1, col2 = st.columns(2)

        with col1:
            nitrogen = st.number_input(
                'Nitrogen (N) [mg/kg]', 
                min_value=20.0, max_value=300.0, step=1.0, format="%.2f", 
                help="Masukkan kadar nitrogen dalam tanah (20 - 298 mg/kg)"
            )

            phosphor = st.number_input(
                'Fosfor (P) [mg/kg]', 
                min_value=25.0, max_value=160.0, step=1.0, format="%.2f", 
                help="Masukkan kadar fosfor dalam tanah (25 - 150 mg/kg)"
            )

            kalium = st.number_input(
                'Kalium (K) [mg/kg]', 
                min_value=30.0, max_value=600.0, step=1.0, format="%.2f", 
                help="Masukkan kadar kalium dalam tanah (30 - 599 mg/kg)"
            )

        with col2:
            ph = st.slider(
                'pH Tanah', 
                min_value=5.0, max_value=8.5, step=0.1, value=5.0,
                help="Masukkan nilai keasaman tanah (rentang 5.01 - 8.20)"
            )

            temp = st.slider(
                'Suhu (°C)', 
                min_value=10, max_value=30, step=1, value=10,
                help="Masukkan suhu udara rata-rata lokasi (10 - 30 °C)"
            )

            humidity = st.slider(
                'Kelembapan Udara (%)', 
                min_value=0, max_value=90, step=1, value=0,
                help="Masukkan kelembapan udara rata-rata (0 - 89%)"
            )

            rainfall = st.slider(
                'Curah Hujan (mm/tahun)', 
                min_value=175, max_value=4500, step=10, value=175,
                help="Masukkan curah hujan rata-rata tahunan (175 - 4452 mm)"
            )

        st.markdown("---")

        model_pilih = st.selectbox(
            '**Pilih Model Klasifikasi**', 
            ('Naïve Bayes', 'Decision Tree'),
            help="Pilih metode klasifikasi yang ingin digunakan"
        )

        submit_button = st.form_submit_button("Prediksi Tanaman")

        if submit_button:
            input_data = np.array([[nitrogen, phosphor, kalium, temp, humidity, ph, rainfall]])
            scaled_data = scaler.transform(input_data)

            if model_pilih == "Naïve Bayes":
                prediction = nb_model.predict(scaled_data)
            else:
                prediction = dt_model.predict(scaled_data)

            pred_label = le.inverse_transform(prediction)[0]

            st.success(f"✅ Tanaman yang direkomendasikan: **{pred_label}**")

            # Format nama file gambar
            formatted_label = pred_label.lower().replace(" ", "")
            img_path = f"image/{formatted_label}.png"

            st.image(img_path, caption=f"Gambar Tanaman: {pred_label}", width=150)
            
                        # Simpan riwayat
            if 'riwayat_prediksi' not in st.session_state:
                st.session_state.riwayat_prediksi = []

            st.session_state.riwayat_prediksi.append({
                "Nitrogen": nitrogen,
                "Fosfor": phosphor,
                "Kalium": kalium,
                "pH": ph,
                "Suhu": temp,
                "Kelembapan": humidity,
                "Curah Hujan": rainfall,
                "Model": model_pilih,
                "Tanaman": pred_label
            })
        # Tampilkan riwayat jika ada
    if 'riwayat_prediksi' in st.session_state and st.session_state.riwayat_prediksi:
        st.markdown("### Riwayat Prediksi")
        df_riwayat = pd.DataFrame(st.session_state.riwayat_prediksi)
        st.dataframe(df_riwayat, use_container_width=True)
        def create_pdf(dataframe):
            pdf = FPDF(orientation='P', unit='mm', format='A4')
            pdf.set_auto_page_break(auto=True, margin=7)
            pdf.add_page()
            pdf.set_font("Arial", size=9)
            pdf.set_left_margin(8)  # ➕ Margin kiri
            pdf.set_right_margin(8)  # ➕ Margin kanan
            pdf.cell(0, 8, txt="Riwayat Prediksi Tanaman", ln=True, align='C')
            pdf.ln(5)
            # Hitung lebar kolom berdasarkan lebar halaman
            page_width = pdf.w - pdf.l_margin - pdf.r_margin
            col_count = len(dataframe.columns)
            col_width = page_width / col_count
            row_height = 10
            # Header kolom
            pdf.set_fill_color(144, 238, 144)  # Light Green
            for col in dataframe.columns:
                pdf.cell(col_width, row_height, str(col), border=1, fill=True, align='C')
            pdf.ln()
            # Baris data
            for i in range(len(dataframe)):
                for item in dataframe.iloc[i]:
                    pdf.cell(col_width, row_height, str(item), border=1, align='C')
                pdf.ln()               
            # Footer/Watermark setelah tabel
            pdf.ln(5)
            pdf.set_font("Arial", size=9, style='I')
            pdf.set_text_color(150)
            pdf.cell(0, 10, "Develop oleh Roni - Tugas Akhir 2025 | Teknik Informatika", align='C')
            # Export to BytesIO
            pdf_output = pdf.output(dest='S').encode('latin1')
            return io.BytesIO(pdf_output)
        pdf_file = create_pdf(df_riwayat)
        st.download_button(
            label="Unduh Riwayat Prediksi (PDF)",
            data=pdf_file,
            file_name="riwayat_prediksi.pdf",
            mime='application/pdf'
        )

elif selected == "Visualisasi":
    # -----------------------------
    # Halaman Visualisasi
    st.title("Visualisasi Data Tanah & Iklim")
    st.markdown("Dataset ini digunakan untuk **mengklasifikasikan jenis tanaman** berdasarkan parameter tanah dan iklim seperti Nitrogen, Phosphor, Kalium, suhu, kelembaban, pH, dan curah hujan.")

    st.info(" Klik kolom pada tabel di bawah untuk melihat detail interaktif. Dataset terdiri dari beberapa fitur dan label jenis tanaman.")

    # Load dataset
    df = pd.read_csv("dataset/data_tanaman.csv")
    df['Label'] = df['Label'].str.title()  # Rapikan kapitalisasi nama tanaman

    # Tampilkan dataset dalam ekspander
    with st.expander("***Lihat Data Awal (klik untuk membuka)***"):
        st.dataframe(df, use_container_width=True, height=400)

        # Tombol untuk mengunduh dataset sebagai file CSV
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label=" Unduh Data sebagai CSV",
            data=csv,
            file_name='data_tanaman.csv',
            mime='text/csv'
        )
        
    # Layout Grid: Distribusi dan Pie Chart
    # -------------------------
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Distribusi Jenis Tanaman")
        distribusi = df['Label'].value_counts().reset_index()
        distribusi.columns = ['Tanaman', 'Jumlah']
        fig_bar = px.bar(
            distribusi, x='Tanaman', y='Jumlah',
            color='Tanaman',
            template='plotly_dark',  # Ganti jadi dark template
            color_discrete_sequence=px.colors.qualitative.Plotly,  # Palet warna cerah & gelap
            title="Jumlah Data per Jenis Tanaman"
        )
        fig_bar.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',  # Buat background plot transparan
            paper_bgcolor='rgba(0,0,0,0)',  # Buat background keseluruhan transparan
            font=dict(color='white')  # Bikin tulisan warna putih supaya jelas di gelap
        )
        st.plotly_chart(fig_bar, use_container_width=True)

        with col2:
            st.subheader("Proporsi Data (Pie Chart)")
            
            fig_pie = px.pie(
                distribusi,
                names='Tanaman',
                values='Jumlah',
                hole=0.2,
                template='plotly_dark'
            )
            
            fig_pie.update_traces(
                textinfo='percent+label',
                textposition='outside',                      # Label di luar potongan
                insidetextorientation='radial',
                marker=dict(line=dict(color='black', width=1)),  # Garis tepi slice
                textfont=dict(size=12),                      # Ukuran teks agar lebih seragam
                pull=[0.03] * len(distribusi)                # Tarik slice sedikit agar tidak padat
            )
            
            fig_pie.update_layout(
                showlegend=True,
                legend_title_text='Jenis Tanaman',
                margin=dict(t=50, b=20, l=20, r=20),
                font=dict(color='black'),
                uniformtext_minsize=10,
                uniformtext_mode='hide',                     # Hindari label tumpang tindih
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            
            st.plotly_chart(fig_pie, use_container_width=True)      
    st.markdown("---")
  
    # -------------------------
    # Scatter Plot Interaktif
    # -------------------------
    st.subheader(" Scatter Plot Antar Variabel")
    
    fitur_x = st.selectbox("Pilih Fitur (Sumbu X)", df.columns[:-1], key="scatter_x")
    fitur_y = st.selectbox("Pilih Fitur (Sumbu Y)", df.columns[:-1], index=1, key="scatter_y")

    fig_scatter = px.scatter(df, x=fitur_x, y=fitur_y,
                             color='Label',
                             symbol='Label',
                             template='plotly_white',
                             title=f"Scatter Plot: {fitur_x} vs {fitur_y}")
    st.plotly_chart(fig_scatter, use_container_width=True)
    st.markdown("---")
    
    # Daftar fitur numerik
    features = ['N', 'P', 'K', 'Temperature', 'Humidity', 'pH', 'Rainfall']

    # Hitung korelasi antar fitur numerik
    corr_matrix = df[features].corr(method='pearson')

    # Buat dua kolom untuk grid tampilan
    col1, col2 = st.columns(2)

    with col1:
        st.subheader(" Heatmap Korelasi Fitur Numerik")
        fig_heatmap = px.imshow(
            corr_matrix,
            text_auto=".2f",
            color_continuous_scale='RdBu',
            origin='upper',
            aspect="auto",
            labels=dict(color="Koefisien Korelasi"),
            title="Heatmap Korelasi Fitur"
        )
        fig_heatmap.update_layout(
            margin=dict(l=20, r=20, t=50, b=20),
            width=700,    
            height=400    
        )
        st.plotly_chart(fig_heatmap, use_container_width=True)

    with col2:
        st.subheader(" Korelasi Tertinggi antar Fitur")
        corr_pairs = corr_matrix.unstack().sort_values(ascending=False)
        corr_pairs = corr_pairs[corr_pairs.index.get_level_values(0) != corr_pairs.index.get_level_values(1)]
        top_corr_df = corr_pairs.drop_duplicates().reset_index()
        top_corr_df.columns = ["Fitur 1", "Fitur 2", "Korelasi"]
        top_corr_df = top_corr_df.head(8)  # Batasi 10 teratas

        # Render dataframe HTML dengan clickable link, st.dataframe() tidak mendukung HTML, gunakan st.markdown
        st.markdown(
            top_corr_df.to_html(escape=False, index=False),
            unsafe_allow_html=True
        )
    

elif selected == "Evaluasi":
    st.title(" Evaluasi Model Machine Learning")
    st.markdown("Evaluasi dilakukan pada dua model klasifikasi: **Naive Bayes** dan **Decision Tree**.")
    
    selected_model = st.selectbox("Pilih Model untuk Evaluasi", ["Naive Bayes", "Decision Tree"])
    
    df_final = pd.read_csv('dataset/data_final.csv')
    X = df_final.drop('Label', axis=1)
    y = df_final['Label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

    if selected_model == "Naive Bayes":
        model = GaussianNB()
        model.fit(X_train, y_train)
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        cv_scores = cross_val_score(model, X, y, cv=StratifiedKFold(10, shuffle=True, random_state=42), scoring='accuracy')

        # Simpan juga hasil Decision Tree untuk perbandingan
        param_grid = {'max_depth': [5, 10, 15], 'min_samples_split': [2, 5], 'min_samples_leaf': [1, 3]}
        grid = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid, cv=5, scoring='accuracy')
        grid.fit(X_train, y_train)

    else:
        param_grid = {'max_depth': [5, 10, 15], 'min_samples_split': [2, 5], 'min_samples_leaf': [1, 3]}
        grid = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid, cv=5, scoring='accuracy')
        grid.fit(X_train, y_train)
        model = grid.best_estimator_
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        cv_scores = cross_val_score(model, X, y, cv=StratifiedKFold(10, shuffle=True, random_state=42), scoring='accuracy')

    # Metrik evaluasi
    metrics_train = {
        "Akurasi": accuracy_score(y_train, y_pred_train),
        "Presisi": precision_score(y_train, y_pred_train, average='weighted', zero_division=0),
        "Recall": recall_score(y_train, y_pred_train, average='weighted', zero_division=0),
        "F1-Score": f1_score(y_train, y_pred_train, average='weighted', zero_division=0)
    }
    metrics_test = {
        "Akurasi": accuracy_score(y_test, y_pred_test),
        "Presisi": precision_score(y_test, y_pred_test, average='weighted', zero_division=0),
        "Recall": recall_score(y_test, y_pred_test, average='weighted', zero_division=0),
        "F1-Score": f1_score(y_test, y_pred_test, average='weighted', zero_division=0)
    }

    # Gabungkan metrik train dan test dalam satu DataFrame
    df_metrics = pd.DataFrame({
        "Metrik": list(metrics_train.keys()),
        "Data Latih (%)": [v * 100 for v in metrics_train.values()],
        "Data Uji (%)": [v * 100 for v in metrics_test.values()]
    })

    # Format angka 2 desimal
    df_metrics["Data Latih (%)"] = df_metrics["Data Latih (%)"].map("{:.2f}".format)
    df_metrics["Data Uji (%)"] = df_metrics["Data Uji (%)"].map("{:.2f}".format)

    st.subheader(" Metrik Evaluasi")
    st.table(df_metrics)

    st.subheader(" Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred_test)
    fig_cm = ff.create_annotated_heatmap(
        z=cm,
        x=sorted(y.unique()),
        y=sorted(y.unique()),
        colorscale='Blues',
        showscale=True
    )
    st.plotly_chart(fig_cm, use_container_width=True)

    # Tampilkan Cross-Validation dengan metric cards
    st.subheader(" Cross-Validation (10-fold)")
    col1, col2 = st.columns(2)
    col1.metric("Rata-rata Akurasi", f"{cv_scores.mean()*100:.2f}%")
    col2.metric("Standar Deviasi", f"{cv_scores.std()*100:.2f}%")

    # Tambahkan boxplot cross-validation scores untuk visualisasi distribusi
    fig_cv = px.box(
        x=cv_scores * 100,
        labels={"x": "Akurasi (%)"},
        title="Distribusi Akurasi Cross-Validation"
    )

    # Grafik Perbandingan Akurasi Model
    accuracy_nb = accuracy_score(y_test, GaussianNB().fit(X_train, y_train).predict(X_test))
    accuracy_dt = accuracy_score(y_test, grid.best_estimator_.predict(X_test))

    akurasi_model = {
        "Naive Bayes": accuracy_nb * 100,
        "Decision Tree": accuracy_dt * 100
    }

    fig_bar = go.Figure()

    fig_bar.add_trace(go.Bar(
        x=list(akurasi_model.keys()),
        y=list(akurasi_model.values()),
        text=[f"{v:.2f}%" for v in akurasi_model.values()],
        textposition='auto',
        marker_color=['skyblue', 'orange'],
        hovertemplate='%{x}<br>Akurasi: %{y:.2f}%<extra></extra>'
    ))

    fig_bar.update_layout(
        title=" Perbandingan Akurasi Model",
        yaxis=dict(title='Akurasi (%)', range=[0, 110]),
        xaxis=dict(title='Model'),
        margin=dict(l=40, r=40, t=60, b=40),
        template='plotly_white',
        hovermode="x unified"
    )

    st.plotly_chart(fig_bar, use_container_width=True)

    st.subheader(" Penjelasan Singkat Model")
    if selected_model == "Naive Bayes":
        st.markdown("""
        **Naive Bayes** adalah model probabilistik sederhana namun efektif.  
        - ✅ Kelebihan: Cepat, ringan, cocok untuk data besar  
        - ❌ Kekurangan: Asumsi independensi antar fitur kadang tidak realistis
        """)
    else:
        st.markdown("""
        **Decision Tree** adalah model berbasis pohon keputusan.  
        - ✅ Kelebihan: Mudah diinterpretasi, fleksibel  
        - ❌ Kekurangan: Rentan overfitting jika tidak di-pruning
        """)

    st.subheader(" Unduh Laporan Evaluasi")

    def generate_pdf(model_name, metrics_train, metrics_test, cv_scores):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(0, 10, f"Laporan Evaluasi Model: {model_name}", ln=True, align='C')
        pdf.ln(5)
        pdf.cell(0, 10, "Data Latih", ln=True)
        for m, v in metrics_train.items():
            pdf.cell(0, 8, f"{m}: {v*100:.2f}%", ln=True)
        pdf.ln(5)
        pdf.cell(0, 10, "Data Uji", ln=True)
        for m, v in metrics_test.items():
            pdf.cell(0, 8, f"{m}: {v*100:.2f}%", ln=True)
        pdf.ln(5)
        pdf.cell(0, 10, "Cross-Validation", ln=True)
        pdf.cell(0, 8, f"Rata-rata Akurasi: {cv_scores.mean()*100:.2f}%", ln=True)
        pdf.cell(0, 8, f"Standar Deviasi: {cv_scores.std()*100:.2f}%", ln=True)
        pdf.ln(5)
        pdf.set_font("Arial", size=9, style='I')
        pdf.set_text_color(150)
        pdf.cell(0, 10, "Develop oleh Roni - Tugas Akhir 2025 | Teknik Informatika", align='C')

        return pdf.output(dest='S').encode('latin1')

    pdf_bytes = generate_pdf(selected_model, metrics_train, metrics_test, cv_scores)
    b64 = base64.b64encode(pdf_bytes).decode()
    href = f'<a href="data:application/octet-stream;base64,{b64}" download="evaluasi_{selected_model.lower().replace(" ", "_")}.pdf">' \
           f'<button style="background-color:green; color:white; padding:10px; border:none; border-radius:5px;">📄 Download PDF</button></a>'
    st.markdown(href, unsafe_allow_html=True)

elif selected == "Tentang":
    st.title(" Tentang Aplikasi Klasifikasi Tanaman")

    st.markdown("""
    <style>
        .highlight {
            background: linear-gradient(90deg, #2b9348 0%, #96c93d 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: 700;
            font-size: 1.3rem;
        }
        .section-title {
            font-size: 1.2rem;
            margin-top: 20px;
            margin-bottom: 10px;
            color: #2b9348;
            font-weight: 700;
        }
        .biodata {
            font-size: 16px;
            line-height: 1.6;
        }
        .info-link a {
            text-decoration: none;
            color: #2b9348;
            font-weight: 600;
        }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<p class="highlight"> Deskripsi Singkat</p>', unsafe_allow_html=True)
    st.write("""
    Aplikasi **Klasifikasi Jenis Tanaman** ini dirancang untuk membantu pengguna menentukan jenis tanaman yang sesuai berdasarkan data **parameter tanah dan iklim**.  
    Dua algoritma utama digunakan, yaitu **Naive Bayes** dan **Decision Tree**, yang telah terbukti akurat dan mudah diinterpretasikan.
    """)

    st.markdown('<p class="highlight"> Fitur Utama</p>', unsafe_allow_html=True)
    st.markdown("""
    -  Prediksi tanaman berdasarkan data input secara cepat dan akurat  
    -  Visualisasi interaktif: Confusion Matrix, Cross-validation, Grafik Akurasi  
    -  Unduh hasil evaluasi dan prediksi dalam bentuk PDF  
    -  Antarmuka Streamlit yang sederhana dan responsif  
    """)

    st.markdown('<p class="highlight"> Tentang Algoritma Machine Learning</p>', unsafe_allow_html=True)
    st.markdown("""
    - **Naive Bayes**  
      Cocok untuk klasifikasi cepat dengan asumsi independensi antar fitur.  
      _Kelebihan:_ Sederhana dan efisien.  
      _Kekurangan:_ Asumsi fitur independen seringkali tidak terpenuhi.

    - **Decision Tree**  
      Algoritma berbasis pohon yang sangat mudah dipahami secara visual.  
      _Kelebihan:_ Mudah diinterpretasi dan fleksibel.  
      _Kekurangan:_ Rentan terhadap overfitting tanpa pruning.
    """)

    st.markdown('<p class="highlight"> Profil Pengembang</p>', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 2])
    with col1:
        st.image("image/roni.jpeg", use_container_width=True)

    with col2:
        # Tabel biodata
        st.markdown("""
            <style>
                .biodata-table td {
                    padding: 6px 14px;
                    vertical-align: top;
                }
                .biodata-table td:first-child {
                    font-weight: 600;
                    color: #333;
                    width: 140px;
                }

                .cv-button {
                    display: inline-block;
                    padding: 12px 20px;
                    margin-top: 15px;
                    font-size: 14px;
                    background-color: #2b9348;
                    border-radius: 6px;
                }
                .cv-button:hover {
                    background-color: #228b22;
                }

                .cv-text {
                    color: white;
                    font-weight: bold;
                    text-decoration: none;
                }
                .biodata-table a {
                    color: #2b9348;
                    text-decoration: none;
                    font-weight: 600;
                }
            </style>

            <table class="biodata-table">
                <tr><td>Nama</td><td>Roni</td></tr>
                <tr><td>NIM</td><td>211220108</td></tr>
                <tr><td>Program Studi</td><td>Teknik Informatika</td></tr>
                <tr><td>Universitas</td><td>Universitas Muhammadiyah Pontianak</td></tr>
                <tr><td>Email</td><td><a href="mailto:ronn.7ex@gmail.com">Email Roni</a></td></tr>
                <tr><td>GitHub</td><td><a href="https://github.com/ronirn" target="_blank">Github Roni</a></td></tr>
                <tr><td>Dribbble</td><td><a href="https://dribbble.com/RONI_ANSYAH" target="_blank">Dribbble Roni</a></td></tr>
            </table>
        """, unsafe_allow_html=True)

        # Tombol CV
        st.markdown("""
            <a href="https://drive.google.com/file/d/1BWpKthPXtkSIZ_yjRJSwA6qSBd5H1ypW/view?usp=sharing" 
            target="_blank" class="cv-button">
                <span class="cv-text">📄 Lihat CV</span>
            </a>
            <br><br>
            <a href="https://forms.gle/FMoaNBnU2JnksTvX7"
            target="_blank" class="cv-button">
                <span class="cv-text">📝 Beri Tanggapan (Kuesioner)</span>
            </a>          
        """, unsafe_allow_html=True)
st.markdown("---")
st.caption("© 2025 Aplikasi Klasifikasi Tanaman | Dibuat oleh Roni | Tugas Akhir Teknik Informatika")

