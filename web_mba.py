import streamlit as st
import pandas as pd
import mlxtend.frequent_patterns.association_rules as association_rules
import mlxtend.frequent_patterns.apriori as apriori

    st.title("Market Basket Analysis UD. Kurnia")
uploaded_file = st.file_uploader("Silakan unggah file transaksi (.csv)", type="csv")

# Jika ada file yang diunggah
if uploaded_file is not None:
    # Baca file menggunakan pandas
    df = pd.read_csv(uploaded_file)
    st.success("Data berhasil diunggah!")

    # Memilih hanya kolom yang dibutuhkan
    df_selected = df.iloc[:, [0, 1, 2, 3, 4, 5]]

    # Mengubah tipe data kolom 'pcs' agar mendukung angka desimal (float)
    df_selected['pcs'] = pd.to_numeric(df_selected['pcs'], errors='coerce')
    cleaned = df_selected.dropna()

    # Langkah 1: Ubah tipe data dari float ke int
    cleaned['date'] = cleaned['date'].astype(int)

    # Langkah 2: Ubah ke string
    cleaned['date'] = cleaned['date'].astype(str)

    # Langkah 3: Format ulang menjadi YYYYMMDD
    def format_date(date_str):
        if len(date_str) == 6:
            year = '20' + date_str[:2]  # Tahun 2000-an
            month_day = date_str[2:]     # Bulan dan hari
            return year + month_day
        return date_str  # Kembalikan nilai asli jika tidak sesuai

    cleaned['date'] = cleaned['date'].apply(format_date)

    # Langkah 4: Konversi ke tipe data tanggal
    cleaned['date'] = pd.to_datetime(cleaned['date'], format='%Y%m%d', errors='coerce')

    cleaned['-'] = cleaned['-'].astype(int)
    cleaned['kode_barang'] = cleaned['kode_barang'].astype(int)
    cleaned['TRX_ID'] = cleaned['TRX_ID'].astype(int)
    cleaned['nama_barang'] = cleaned['nama_barang'].astype(str)

    df_sorted = cleaned.groupby('TRX_ID', group_keys=False).apply(lambda x: x.sort_values('nama_barang'))
    trx_count = len(df_sorted)
    item_count = len(pd.unique(df_sorted['nama_barang']))
    txid_count = df_sorted['TRX_ID'].nunique()


    df_sorted['year_month'] = df_sorted['date'].dt.to_period('M')

    # Dictionary untuk menyimpan hasil
    item_counts = {}

    # Looping setiap baris untuk menghitung kemunculan item
    for index, row in df_sorted.iterrows():
        year_month = row['year_month']
        items = row['nama_barang'].split(', ')  # Pisahkan item jika lebih dari satu

        for item in items:
            key = (year_month, item)  # Gabungan Tahun-Bulan dan Item sebagai kunci
            item_counts[key] = item_counts.get(key, 0) + 1  # Tambah jumlah kemunculan

    # Ubah hasil ke DataFrame
    result_df = pd.DataFrame([(ym, item, count) for (ym, item), count in item_counts.items()],
                            columns=['year_month', 'nama_barang', 'total_kemunculan'])

    # Urutkan berdasarkan Tahun-Bulan dan total kemunculan terbesar
    result_df = result_df.sort_values(by=['year_month', 'total_kemunculan'], ascending=[True, False])


    # TOP 5 TRANSACTION
    count = df_sorted['nama_barang'].value_counts().reset_index()
    count.columns = ['nama_barang', 'jumlah_transaksi']

    # Mengambil 5 produk dengan jumlah transaksi tertinggi
    top_5 = count.nlargest(5, 'jumlah_transaksi')
    # Agar cocok dengan st.bar_chart, set index ke nama barang
    top_5_chart = top_5.set_index('nama_barang')


    # STYLE
    # CSS custom untuk styling kartu
    st.markdown("""
        <style>
        .card {
            background-color: #f4efe3;
            padding: 30px;
            border-radius: 10px;
            text-align: center;
            box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
            margin: 10px;
        }
        .card-number {
            color: #f45b22;
            font-size: 30px;
            font-weight: bold;
            margin-bottom: 5px;
        }
        .card-label {
            color: #1e3a5f;
            font-size: 14px;
            font-weight: normal;
            letter-spacing: 1px;
        }
        </style>
    """, unsafe_allow_html=True)


    with st.sidebar:
        st.title("About")
        st.markdown(
            """
            <div style="border-radius: 10px; padding: 10px; background-color: #e6e6fa; color: #80A6D0">
                Desain sistem Market Basket Analysis ini dirancang untuk membantu pemilik toko bangunan UD. Kurnia dalam memahami pola pembelian konsumen berdasarkan data transaksi yang telah diinput. Sistem ini dibangun menggunakan algoritma Apriori dengan pilihan parameter minimum support sebesar 0.05, 0.02, 0.005, dan 0.001 dan pilihan minimum threshold sebesar 1, 1.5, dan 2. Dengan tampilan yang informatif dan mudah dipahami, sistem ini diharapkan dapat memberikan wawasan yang berguna dalam pengambilan keputusan, khususnya untuk pengelolaan stok yang lebih optimal dan menemukan aturan asosiasi antar produk yang sering dibeli bersamaan.
            </div>
            """,
            unsafe_allow_html=True
        )

    # VIEW
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(f"""
            <div class="card">
                <div class="card-number">{txid_count}</div>
                <div class="card-label">TOTAL TRANSAKSI</div>
            </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
            <div class="card">
                <div class="card-number">{trx_count}</div>
                <div class="card-label">TOTAL DATA</div>
            </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
            <div class="card">
                <div class="card-number">{item_count}</div>
                <div class="card-label">TOTAL ITEM</div>
            </div>
        """, unsafe_allow_html=True)


    # Tampilkan judul
    st.markdown(
        """
        <div style="margin-top: 50px;">
        </div>
        """,
        unsafe_allow_html=True
    )
    st.subheader("📊 5 Produk dengan Jumlah Transaksi Tertinggi")
    st.bar_chart(top_5_chart)


    # SELECT TOP TRXs PER MONTH
    # Ambil 5 item terbanyak di setiap bulan
    top_trxs_per_month = result_df.groupby('year_month').apply(lambda x: x.nlargest(5, 'total_kemunculan')).reset_index(drop=True)
    months = []

    for month in top_trxs_per_month['year_month'].unique():
        months.append(month)

    def getTopTrxPerMonth(month):
        subset = top_trxs_per_month[top_trxs_per_month['year_month'] == month]
        st.bar_chart(subset, x='nama_barang', x_label="Nama Barang", y='total_kemunculan', y_label="Jumlah Kemunculan", horizontal=True)
        return month

    st.markdown(
        """
        <div style="margin-top: 50px;">
        </div>
        """,
        unsafe_allow_html=True
    )
    st.subheader("📊 5 Produk dengan Jumlah Transaksi Tertinggi (Bulan)")
    topTrxPerMonthOptions = st.selectbox(
        "Pilih bulan:",
        months,
        key="topTrx"
    )

    st.write(getTopTrxPerMonth(topTrxPerMonthOptions))
    # END SELECT TOP TRXs PER MONTH

    # SELECT TOP SALES PER MONTH
    # Ambil format tahun-bulan
    df_sorted["tahun_bulan"] = df_sorted["date"].dt.to_period("M").astype(str)
    # Hitung jumlah pcs per tahun-bulan dan nama_barang
    monthly_sales = df_sorted.groupby(["tahun_bulan", "nama_barang"]).agg({"pcs": "sum"}).reset_index()
    # Sortir data berdasarkan tahun-bulan dan jumlah pcs yang terjual
    monthly_sales = monthly_sales.sort_values(by=["tahun_bulan", "pcs"], ascending=[True, False])
    # Ambil 5 item terbanyak di setiap bulan
    top_saless_per_month = monthly_sales.groupby('tahun_bulan').apply(lambda x: x.nlargest(5, 'pcs')).reset_index(drop=True)

    topSalesMonths = []

    for month in top_saless_per_month['tahun_bulan'].unique():
        topSalesMonths.append(month)

    def getTopSalesPerMonth(month):
        subset = top_saless_per_month[top_saless_per_month['tahun_bulan'] == month]
        st.bar_chart(subset, x='nama_barang', x_label="Nama Barang", y='pcs', y_label="Jumlah Kemunculan", horizontal=True)
        return month

    st.markdown(
        """
        <div style="margin-top: 50px;">
        </div>
        """,
        unsafe_allow_html=True
    )
    st.subheader("📊 5 Produk dengan Jumlah Item yang Terjual (Bulan)")
    topSalesPerMonthOptions = st.selectbox(
        "Pilih bulan:",
        topSalesMonths,
        key="topSales"
    )

    st.write(getTopSalesPerMonth(topSalesPerMonthOptions))
    # END SELECT TOP SALES PER MONTH

    # VIEW
    st.title("Penerapan Algoritma Apriori")

    # ASSOCIATION RULES
    df1 = df_sorted.set_index('TRX_ID')
    df1['date'] = pd.to_datetime(df1['date']).dt.strftime('%Y%m%d').astype(int)
    df1HotEncoded = df1.pivot_table(index='TRX_ID', columns='nama_barang', values='pcs', aggfunc='sum').fillna(0)
    df1HotEncoded[df1HotEncoded > 0] = 1
    dfSparse = df1HotEncoded.astype(pd.SparseDtype(int, fill_value=0))

    # Pilih nilai minimum support
    min_support = st.selectbox(
        "Pilih nilai Minimum Support:",
        options=[0.05, 0.02, 0.005, 0.001],
        index=0,  # default 0.005
        format_func=lambda x: f"{x:.3f}"
    )

    # Pilih nilai minimum threshold (lift)
    min_threshold = st.selectbox(
        "Pilih nilai Minimum Threshold (Lift):",
        options=[1, 1.5, 2],
        index=0
    )

    df3 = apriori(dfSparse, min_support=min_support, use_colnames=True)
    df4 = association_rules(df3, metric='lift', min_threshold=min_threshold)[['antecedents', 'consequents', 'support', 'confidence', 'lift']]

    st.markdown(
        """
        <div style="margin-top: 50px;">
        </div>
        """,
        unsafe_allow_html=True
    )
    st.subheader("⭐ Association Rules")
    # Buat pilihan format string "if [A] then [B]"
    options = df4.apply(lambda row: f"Jika membeli {', '.join(row['antecedents'])}, maka akan membeli  {', '.join(row['consequents'])} juga.", axis=1)
    selected = st.selectbox("Pilih aturan asosiasi:", options, key="rule_selector")

    # Tampilkan nilai confidence dari rule yang dipilih
    selected_row = df4.iloc[options[options == selected].index[0]]
    st.markdown(f"**Nilai Kepercayaan** dari aturan tersebut adalah: `{selected_row['confidence']:.4f}`")
    # END ASSOCIATION RULES


    # ITEM RECOMMENDATION
    st.markdown(
        """
        <div style="margin-top: 50px;">
        </div>
        """,
        unsafe_allow_html=True
    )
    st.subheader("🎯 Item Recommendation")
    # Buat daftar unik dari antecedents
    unique_antecedents = df4['antecedents'].apply(lambda x: ', '.join(sorted(x))).unique()

    # Selectbox untuk memilih item pembelian (antecedents)
    selected_antecedent = st.selectbox("Pilih item yang dibeli:", unique_antecedents, key="rekomendasi_selector")

    # Filter rules dengan antecedents yang cocok
    matched_rules = df4[df4['antecedents'].apply(lambda x: ', '.join(sorted(x))) == selected_antecedent]

    # Ambil consequents sebagai rekomendasi
    recommendations = matched_rules['consequents'].apply(lambda x: ', '.join(x)).unique()

    # Tampilkan rekomendasi
    if len(recommendations) > 0:
        st.write(f"📦 Rekomendasi produk berdasarkan item '{selected_antecedent}' yang dibeli:")
        for item in recommendations:
            st.markdown(f"- {item}")
    else:
        st.warning("Tidak ada rekomendasi untuk item tersebut.")
    # END ITEM RECOMMENDATION
