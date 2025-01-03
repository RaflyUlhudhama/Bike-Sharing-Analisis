import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from datetime import datetime
import streamlit as st
from streamlit_folium import st_folium

# Memuat dataset
day_data_path = 'dashboard/day.csv'
hour_data_path = 'dashboard/hour.csv'

day_data = pd.read_csv(day_data_path)
hour_data = pd.read_csv(hour_data_path)

# Konversi tanggal ke format datetime
day_data['dteday'] = pd.to_datetime(day_data['dteday'])
hour_data['dteday'] = pd.to_datetime(hour_data['dteday'])

# Dashboard Streamlit
st.set_page_config(page_title="Bike-Sharing Analysis Dashboard", layout="wide", page_icon="🚴")

# Header
st.title("🚴 Bike-Sharing Analysis Dashboard")
st.markdown("### Data Exploratory Analysis, RFM Analysis, and Visualization")

# Tab Layout
tabs = st.tabs(["Overview", "EDA", "RFM Analysis", "Kesimpulan"])

# Tab 1: Overview
with tabs[0]:
    st.header("Dataset Overview")
    st.write("**Data Harian (Day Data):**")
    st.dataframe(day_data.head(), use_container_width=True)
    st.caption("Dataset harian memuat informasi total peminjaman sepeda, kondisi cuaca, dan informasi temporal lainnya.")

    st.write("**Data Per Jam (Hourly Data):**")
    st.dataframe(hour_data.head(), use_container_width=True)
    st.caption("Dataset per jam menyediakan informasi lebih rinci terkait peminjaman sepeda dalam satu hari.")

    st.subheader("Informasi Data")
    st.write("**Informasi Data Harian:**")
    buffer = pd.DataFrame(day_data.dtypes, columns=["Tipe Data"])
    buffer["Nilai Kosong"] = day_data.isnull().sum()
    st.dataframe(buffer)
    st.caption("Dataset harian tidak memiliki nilai kosong, menunjukkan data siap untuk dianalisis.")

    st.write("**Informasi Data Per Jam:**")
    buffer = pd.DataFrame(hour_data.dtypes, columns=["Tipe Data"])
    buffer["Nilai Kosong"] = hour_data.isnull().sum()
    st.dataframe(buffer)
    st.caption("Dataset per jam juga bebas dari nilai kosong, menjamin kualitas data yang baik.")

# Tab 2: EDA
with tabs[1]:
    st.header("Exploratory Data Analysis (EDA)")
    
    # Distribusi Total Peminjaman Sepeda
    st.subheader("Distribusi Total Peminjaman Sepeda")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(day_data['cnt'], kde=True, bins=30, color='blue', ax=ax)
    ax.set_title('Distribusi Total Peminjaman Sepeda (cnt)')
    ax.set_xlabel('Total Peminjaman')
    ax.set_ylabel('Frekuensi')
    st.pyplot(fig)
    st.caption("Distribusi total peminjaman sepeda menunjukkan pola berbentuk lonceng dengan mayoritas peminjaman berkisar antara 3000 hingga 5000 sepeda per hari.")

    # Heatmap Korelasi
    st.subheader("Heatmap Korelasi Data Harian")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(day_data.corr(), annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
    ax.set_title('Heatmap Korelasi - Data Harian')
    st.pyplot(fig)
    st.caption("Heatmap korelasi menunjukkan hubungan yang kuat antara suhu (temp) dan total peminjaman (cnt). Variabel cuaca lain seperti hum (kelembapan) juga memiliki korelasi yang moderat.")

# Tab 3: RFM Analysis
with tabs[2]:
    st.header("RFM Analysis")
    hari_ini = day_data['dteday'].max()
    rfm = day_data.groupby('weekday').agg({
        'dteday': lambda x: (hari_ini - x.max()).days,
        'cnt': ['sum', 'count']
    }).reset_index()

    rfm.columns = ['Weekday', 'Recency', 'Monetary', 'Frequency']
    st.write("**RFM Metrics:**")
    st.dataframe(rfm, use_container_width=True)
    st.caption("Analisis RFM membantu memahami pola peminjaman berdasarkan hari dalam seminggu. Recency menunjukkan waktu sejak peminjaman terakhir, Monetary adalah total peminjaman, dan Frequency adalah jumlah kejadian peminjaman.")

    # Visualisasi Distribusi Total Peminjaman per Hari
    st.subheader("Distribusi Total Peminjaman Sepeda Berdasarkan Hari")
    day_labels = ['Senin', 'Selasa', 'Rabu', 'Kamis', 'Jumat', 'Sabtu', 'Minggu']
    day_data['weekday'] = day_data['weekday'].replace(range(7), day_labels)
    total_peminjaman = day_data.groupby('weekday')['cnt'].sum().reindex(day_labels)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    total_peminjaman.plot(kind='bar', color='blue', ax=ax)
    ax.set_title('Distribusi Total Peminjaman Sepeda Berdasarkan Hari')
    ax.set_ylabel('Total Peminjaman')
    ax.set_xlabel('Hari')

    # Mengatur tick yang sesuai dengan jumlah hari dalam seminggu
    ax.set_xticks(range(len(day_labels)))  # Set ticks sesuai jumlah hari (7 hari)
    ax.set_xticklabels(day_labels, rotation=45)  # Set label untuk setiap tick

    st.pyplot(fig)
    st.caption("Hari Sabtu dan Minggu memiliki jumlah peminjaman tertinggi, menunjukkan bahwa penggunaan sepeda meningkat saat akhir pekan.")


with tabs[3]:
    st.header("Kesimpulan Analisis")
    
    # Visualisasi Tren Peminjaman Sepeda Berdasarkan Hari
    st.subheader("Tren Peminjaman Sepeda Berdasarkan Hari")
    fig, ax = plt.subplots(figsize=(10, 6))
    total_peminjaman.plot(kind='line', color='orange', ax=ax, marker='o')
    ax.set_title('Tren Peminjaman Sepeda Berdasarkan Hari')
    ax.set_ylabel('Total Peminjaman')
    ax.set_xlabel('Hari')
    ax.set_xticks(range(len(day_labels)))
    ax.set_xticklabels(day_labels, rotation=45)
    st.pyplot(fig)
    st.caption("Grafik ini menunjukkan bahwa peminjaman sepeda lebih tinggi pada akhir pekan (Sabtu dan Minggu) dibandingkan dengan hari kerja, yang mencerminkan pola penggunaan sepeda lebih sering untuk rekreasi saat akhir pekan.")

    # Visualisasi Pengaruh Cuaca
    st.subheader("Pengaruh Cuaca terhadap Peminjaman Sepeda")
    
    # Grafik Suhu vs Peminjaman Sepeda
    st.write("Grafik Suhu (Temp) vs Peminjaman Sepeda")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x=day_data['temp'], y=day_data['cnt'], color='blue', ax=ax)
    ax.set_title('Hubungan Suhu (Temp) dengan Jumlah Peminjaman Sepeda')
    ax.set_xlabel('Suhu (°C)')
    ax.set_ylabel('Jumlah Peminjaman Sepeda')
    st.pyplot(fig)
    st.caption("Grafik ini menunjukkan hubungan yang kuat antara suhu yang lebih tinggi dan peningkatan jumlah peminjaman sepeda, mengindikasikan bahwa orang cenderung bersepeda lebih banyak saat cuaca lebih hangat.")

    # Grafik Kelembapan vs Peminjaman Sepeda
    st.write("Grafik Kelembapan (Hum) vs Peminjaman Sepeda")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x=day_data['hum'], y=day_data['cnt'], color='green', ax=ax)
    ax.set_title('Hubungan Kelembapan (Hum) dengan Jumlah Peminjaman Sepeda')
    ax.set_xlabel('Kelembapan (%)')
    ax.set_ylabel('Jumlah Peminjaman Sepeda')
    st.pyplot(fig)
    st.caption("Grafik ini menunjukkan korelasi negatif antara kelembapan dan peminjaman sepeda. Pada hari dengan kelembapan yang lebih tinggi, peminjaman sepeda cenderung berkurang, mungkin karena kenyamanan bersepeda yang lebih rendah saat udara lembap.")

    # Grafik Kecepatan Angin vs Peminjaman Sepeda
    st.write("Grafik Kecepatan Angin (Windspeed) vs Peminjaman Sepeda")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x=day_data['windspeed'], y=day_data['cnt'], color='red', ax=ax)
    ax.set_title('Hubungan Kecepatan Angin (Windspeed) dengan Jumlah Peminjaman Sepeda')
    ax.set_xlabel('Kecepatan Angin (km/h)')
    ax.set_ylabel('Jumlah Peminjaman Sepeda')
    st.pyplot(fig)
    st.caption("Grafik ini menunjukkan bahwa kecepatan angin yang lebih tinggi cenderung menurunkan jumlah peminjaman sepeda. Hal ini dapat dipengaruhi oleh faktor kenyamanan dan keselamatan bersepeda.")

    # Visualisasi Pola Penggunaan Sepeda berdasarkan Hari
    st.subheader("Distribusi Peminjaman Sepeda Berdasarkan Hari")
    fig, ax = plt.subplots(figsize=(10, 6))
    total_peminjaman.plot(kind='bar', color='purple', ax=ax)
    ax.set_title('Distribusi Peminjaman Sepeda Berdasarkan Hari')
    ax.set_ylabel('Total Peminjaman')
    ax.set_xlabel('Hari')
    st.pyplot(fig)
    st.caption("Grafik ini memperlihatkan bahwa peminjaman sepeda mencapai puncaknya pada akhir pekan, terutama pada hari Sabtu dan Minggu, mencerminkan tingginya aktivitas rekreasi pada waktu tersebut.")

    # Kesimpulan dalam Paragraf
    st.markdown(
        "### **Kesimpulan Analisis**\n"
        "1. **Pengaruh Cuaca (Suhu, Kelembapan, Kecepatan Angin) terhadap Penggunaan Sepeda Harian dan Per Jam**\n"
        "   - **Suhu (temp):** Suhu yang lebih tinggi memiliki korelasi positif yang kuat dengan peningkatan jumlah peminjaman sepeda. Hal ini menunjukkan bahwa orang lebih cenderung menggunakan sepeda ketika cuaca lebih hangat, yang dapat dihubungkan dengan kegiatan luar ruangan yang lebih banyak, seperti bersepeda untuk rekreasi atau transportasi.\n"
        "   - **Kelembapan (hum):** Kelembapan juga berpengaruh, meskipun lebih moderat. Peningkatan kelembapan biasanya berhubungan dengan cuaca yang lebih panas dan lembap, yang dapat mengurangi kenyamanan dalam bersepeda. Oleh karena itu, pada hari dengan kelembapan tinggi, jumlah peminjaman sepeda cenderung sedikit berkurang.\n"
        "   - **Kecepatan Angin (windspeed):** Kecepatan angin yang sangat tinggi mengurangi kenyamanan bersepeda, meskipun pengaruhnya tidak sekuat suhu. Pada hari dengan angin yang sangat kencang, peminjaman sepeda cenderung lebih rendah karena faktor kenyamanan dan keselamatan.\n"
        "\n"
        "2. **Pola Penggunaan Sepeda antara Hari Kerja dan Akhir Pekan**\n"
        "   - **Akhir Pekan (Sabtu dan Minggu):** Hari Sabtu dan Minggu menunjukkan jumlah peminjaman sepeda yang jauh lebih tinggi dibandingkan dengan hari kerja. Peningkatan ini kemungkinan besar disebabkan oleh meningkatnya aktivitas rekreasi, seperti bersepeda untuk olahraga atau hiburan di luar ruangan. Akhir pekan juga sering kali menjadi waktu yang lebih bebas untuk banyak orang, yang memungkinkan mereka untuk menggunakan sepeda lebih sering.\n"
        "   - **Hari Kerja (Senin-Jumat):** Penggunaan sepeda pada hari kerja lebih stabil dan cenderung moderat. Meskipun terdapat fluktuasi pada jam-jam tertentu, secara keseluruhan, peminjaman sepeda di hari kerja lebih banyak dipengaruhi oleh kebutuhan transportasi harian, seperti untuk perjalanan menuju tempat kerja atau sekolah. Hal ini menunjukkan bahwa bersepeda di hari kerja lebih bersifat fungsional dan utilitarian dibandingkan dengan di akhir pekan yang lebih bersifat rekreasi.\n"
        "\n"
        "### **Poin Kesimpulan Utama:**\n"
        "1. **Pengaruh Cuaca:** Suhu yang lebih tinggi meningkatkan jumlah peminjaman sepeda, sementara kelembapan yang lebih tinggi dan kecepatan angin yang kencang dapat mengurangi peminjaman sepeda.\n"
        "2. **Pola Penggunaan Sepeda:** Peminjaman sepeda jauh lebih tinggi pada akhir pekan (Sabtu dan Minggu), yang menunjukkan pola rekreasi yang lebih dominan, sedangkan pada hari kerja, peminjaman lebih moderat dan lebih dipengaruhi oleh kebutuhan transportasi rutin.\n"
        "3. **Strategi Promosi:** Untuk meningkatkan penggunaan sepeda, dapat dipertimbangkan untuk menyelenggarakan promosi berbasis cuaca, seperti diskon pada hari-hari dengan suhu tinggi atau program spesial pada akhir pekan untuk mendorong penggunaan sepeda lebih lanjut.\n"
    )

