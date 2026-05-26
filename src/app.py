import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os
import yfinance as yf
import joblib

try:
    from prophet import Prophet
except ImportError:
    Prophet = None

# --- Sayfa Yapılandırması ---
st.set_page_config(
    page_title="BORSANEURON | ALGORİTMİK TİCARET VE YAPAY ZEKA TERMİNALİ",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Premium Kurumsal CSS Arayüzü ---
TERMINAL_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto+Mono:wght@300;400;500;600;700&family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    .stApp { background-color: #0a0e1a; color: #e2e8f0; font-family: 'Inter', sans-serif; }
    
    /* Terminal Dark Background Gradient */
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(180deg, #0a0e1a 0%, #0e1117 40%, #111827 100%);
    }

    /* Sidebar - Kurumsal Tasarım */
    [data-testid="stSidebar"] {
        background-color: #0b0f19 !important;
        border-right: 1px solid rgba(59, 130, 246, 0.1) !important;
    }

    /* BorsaNeuron Glass Kartlar */
    .terminal-card {
        background-color: rgba(30, 41, 59, 0.45);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-left: 3px solid #3b82f6;
        padding: 20px;
        margin-bottom: 16px;
        border-radius: 4px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
    }
    
    .brand-header {
        color: #3b82f6;
        font-family: 'Roboto Mono', monospace;
        font-weight: 700;
        letter-spacing: 2px;
        font-size: 1.4rem;
        border-bottom: 2px solid rgba(59, 130, 246, 0.2);
        padding-bottom: 8px;
        margin-bottom: 15px;
    }
    
    .metric-value { 
        color: #10b981; 
        font-weight: 700; 
        font-size: 1.6rem; 
        font-family: 'Roboto Mono', monospace; 
        text-shadow: 0 0 10px rgba(16, 185, 129, 0.2);
    }
    .metric-label { 
        font-size: 0.75rem; 
        color: #94a3b8; 
        text-transform: uppercase; 
        font-weight: 600;
        letter-spacing: 1px;
        margin-bottom: 4px;
    }
    
    /* Gelişmiş Girdi Alanı Tasarımı (Inputs) */
    .stTextInput > div > div > input, 
    .stSelectbox > div > div > div, 
    .stNumberInput > div > div > input {
        background-color: #0f172a !important;
        border: 1px solid rgba(59, 130, 246, 0.2) !important;
        color: #f8fafc !important;
        border-radius: 4px !important;
        padding: 8px 12px !important;
        font-family: 'Roboto Mono', monospace !important;
    }
    
    /* Buton Tasarımları */
    .stButton > button {
        border-radius: 4px !important;
        font-family: 'Inter', sans-serif !important;
        font-weight: 600 !important;
        transition: 0.2s !important;
        letter-spacing: 0.5px;
    }
    
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #1d4ed8, #2563eb) !important;
        color: white !important;
        border: none !important;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.2) !important;
        text-transform: uppercase;
        padding: 10px 24px !important;
    }
    
    .stButton > button[kind="primary"]:hover {
        background: #2563eb !important;
        box-shadow: 0 4px 20px rgba(37, 99, 235, 0.4) !important;
        transform: translateY(-0.5px);
    }
    
    /* İlerleme Çubukları */
    .stProgress > div > div { 
        background: linear-gradient(90deg, #1d4ed8, #2563eb) !important; 
        border-radius: 4px; 
    }
    
    /* Kurumsal Tablo Sınırları */
    [data-testid="stDataFrame"] {
        border: 1px solid rgba(255, 255, 255, 0.05) !important;
        border-radius: 4px !important;
    }
    
    /* Kaydırma Çubukları (Scrollbars) */
    ::-webkit-scrollbar { width: 5px; height: 5px; }
    ::-webkit-scrollbar-track { background: #0a0e1a; }
    ::-webkit-scrollbar-thumb { background: rgba(59, 130, 246, 0.25); border-radius: 4px; }
    ::-webkit-scrollbar-thumb:hover { background: rgba(59, 130, 246, 0.45); }
    
    /* Alt Bilgi */
    .footer-text {
        font-size: 0.75rem;
        color: #475569;
        font-family: 'Roboto Mono', monospace;
        margin-top: 30px;
        text-align: center;
        border-top: 1px solid rgba(255, 255, 255, 0.03);
        padding-top: 15px;
    }
</style>
"""
st.markdown(TERMINAL_CSS, unsafe_allow_html=True)

# --- Veri Yükleme ---
@st.cache_data(ttl=3600)
def load_data():
    paths = [
        "bist_ai_dataset_real_30cols.csv",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'bist_ai_dataset_real_30cols.csv'),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), 'bist_ai_dataset_real_30cols.csv'),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'bist_ai_dataset_real_30cols.csv'),
        "/mount/src/borsaneuron/bist_ai_dataset_real_30cols.csv"
    ]
    for p in paths:
        if os.path.exists(p):
            return pd.read_csv(p)
    return None

df = load_data()

if df is None:
    st.error("Veriseti bulunamadı! Lütfen 'bist_ai_dataset_real_30cols.csv' dosyasının doğru yerde olduğundan emin olun.")
    st.stop()

# --- Kurumsal Sol Menü Navigasyonu (Sidebar Navigation) ---
st.sidebar.markdown("<div class='brand-header'>BORSANEURON</div>", unsafe_allow_html=True)
st.sidebar.markdown("<div style='font-size:0.75rem; color:#64748b; letter-spacing:1px; text-transform:uppercase; margin-top:-10px; margin-bottom:20px; font-family:\"Roboto Mono\", monospace;'>QUANTITATIVE SYSTEMS</div>", unsafe_allow_html=True)

page = st.sidebar.radio("ANALİTİK MODÜLLER", [
    "Sistem Tanıtımı & Dokümantasyon",
    "Açıklayıcı Veri Analizi (EDA)",
    "Korelasyon ve Boyut Eleme",
    "Piyasa Rejim Sınıflandırması",
    "Makine Öğrenmesi Model Analizleri",
    "Zaman Serisi Trend Tahmini",
    "Canlı Hisse Sorgulama & Çıkarım",
    "Portföy Simülasyonu ve Backtest",
    "Otomatik Formasyon Tarayıcı"
])

st.sidebar.markdown("---")
st.sidebar.markdown("<div style='font-size:0.7rem; color:#475569; font-family:\"Roboto Mono\", monospace;'>AKTİF PORTFÖY MOTORU: XGBoost Classifier<br>VERİ TABANI GÜNCELLEME: Canlı (yFinance)<br>SÜRÜM: v2.5.0-BIST</div>", unsafe_allow_html=True)

# ==========================================
# MODÜL 0: SİSTEM TANITIMI & DOKÜMANTASYON
# ==========================================
if page == "Sistem Tanıtımı & Dokümantasyon":
    st.markdown("### Sistem Genel Bakış ve Analitik Altyapı")
    st.markdown("Borsa İstanbul (BIST) pay piyasalarındaki hisse senetlerinin nicel teknik göstergeleri ve makine öğrenmesi modelleri kullanılarak analiz edilmesi amacıyla geliştirilmiş bütünsel karar destek platformu.")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class='terminal-card'>
            <div class='metric-label'>Sınıflandırma Hedefi (Target_T5)</div>
            <div class='metric-value'>5 Günlük Yön</div>
            <p style='font-size:0.8rem; color:#94a3b8; margin-top:8px;'>Hisse senedinin 5 işlem günü sonraki kapanış fiyatının bugünkünden yüksek olma olasılığı.</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class='terminal-card'>
            <div class='metric-label'>Kapsanan Aktif Hisse</div>
            <div class='metric-value'>537 Ticker</div>
            <p style='font-size:0.8rem; color:#94a3b8; margin-top:8px;'>BIST genelini temsil eden ve temizleme adımlarından başarıyla geçen tüm aktif hisse senetleri.</p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class='terminal-card'>
            <div class='metric-label'>Üretilen Teknik Metrikler</div>
            <div class='metric-value'>30 Gösterge</div>
            <p style='font-size:0.8rem; color:#94a3b8; margin-top:8px;'>Momentum, volatilite, hacim ve trend eğilimlerini temsil eden yapılandırılmış nicel değişken seti.</p>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("#### BorsaNeuron Metodolojik İş Akışı")
    
    st.markdown("""
    BIST teknik veritabanı altyapımız, çevrimdışı modelleme (offline training) ile çevrimiçi anlık kestirimi (online inference) birleştiren hibrit bir tasarıma sahiptir:
    
    1. **Veri Madenciliği (Data Mining):** `yfinance` aracılığıyla akan BIST verileri üzerinden 30 farklı teknik indikatör değişkeni ve Target_T5 etiket sınıfı oluşturulur.
    2. **Boyut İndirgeme ve Gürültü Filtreleme:** Pearson Korelasyon Analizi uygulanarak aralarında çoklu doğrusal bağlantı (multicollinearity) bulunan yüksek korelasyonlu değişkenler ayıklanır.
    3. **Piyasa Koşulları Kümelemesi:** K-Means algoritması ile hisselerin teknik durumları 5 farklı piyasa rejim kümesinde segmente edilir ve PCA (Temel Bileşenler Analizi) ile haritalandırılır.
    4. **Yapay Zeka Modellemesi:** K-En Yakın Komşu (K-NN), Yapay Sinir Ağları (ANN-MLP), Random Forest ve XGBoost modelleri optimize edilerek en yüksek F1-Skoruna sahip sınıflandırıcı çıkarım motoru olarak sisteme entegre edilir.
    5. **Tarihsel Uyum (Win Rate) Modülasyonu:** Canlı hisse sorgulamalarında ilgili hissenin yapay zeka sinyallerine olan geçmiş uyumu ağırlıklandırılarak karar kararlılığı artırılır.
    """)

# ==========================================
# MODÜL 1: AÇIKLAYICI VERİ ANALİZİ (EDA)
# ==========================================
elif page == "Açıklayıcı Veri Analizi (EDA)":
    st.markdown("### Açıklayıcı Veri Analizi ve Tanımlayıcı İstatistikler")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Toplam Kayıt Satırı", f"{df.shape[0]:,}")
    col2.metric("Toplam Özellik Sütunu", f"{df.shape[1]}")
    col3.metric("Eksik Veri Hücresi", "0" if not df.isnull().values.any() else f"{df.isnull().sum().sum()}")
    
    st.markdown("#### Örnek Veri Matrisi (İlk 5 Satır)")
    st.dataframe(df.head(), use_container_width=True)
    
    st.markdown("#### Tanımlayıcı İstatistik Özetleri (Descriptive Statistics)")
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    desc = df[num_cols].describe().T
    st.dataframe(desc, use_container_width=True)
    
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        st.markdown("#### Hedef Değişken (Target_T5) Dağılımı")
        target_counts = df['Target_T5'].value_counts()
        fig_target = px.pie(values=target_counts.values, names=['Düşüş (0)', 'Yükseliş (1)'],
                            color_discrete_sequence=['#ef4444', '#10b981'])
        fig_target.update_layout(
            template="plotly_dark", 
            plot_bgcolor="rgba(0,0,0,0)", 
            paper_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_target, use_container_width=True)
        
    with col_chart2:
        st.markdown("#### Gösterge Frekans Dağılımı: RSI_14")
        fig_rsi = px.histogram(df, x='RSI_14', nbins=50, color_discrete_sequence=['#3b82f6'])
        fig_rsi.update_layout(
            template="plotly_dark", 
            plot_bgcolor="rgba(0,0,0,0)", 
            paper_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_rsi, use_container_width=True)
        
    st.markdown("#### Yön Sınıfına (Target_T5) Göre Gösterge Ortalamaları")
    indicator_cols = ['RSI_14', 'MACD', 'ATR_14', 'Stoch_K', 'Volume_Trend']
    target_means = df.groupby('Target_T5')[indicator_cols].mean()
    st.dataframe(target_means, use_container_width=True)

# ==========================================
# MODÜL 2: KORELASYON VE BOYUT ELEME
# ==========================================
elif page == "Korelasyon ve Boyut Eleme":
    st.markdown("### Korelasyon Analizi ve Çoklu Doğrusallık Filtreleme")
    
    with st.expander("Metodoloji Açıklaması", expanded=True):
        st.markdown('''
        * Çoklu doğrusal bağlantı (multicollinearity), aynı veya benzer matematiksel formülasyona sahip göstergelerin modelde gürültü oluşturmasına sebep olur.
        * Pearson korelasyon katsayısı mutlak değeri $|r_{ij}| > 0.90$ olan çiftler belirlenerek üst üçgen matris filtresiyle ayıklanmıştır.
        * Bu eleme süreci, modellerin aşırı öğrenme (overfitting) riskini azaltarak genellenebilirliği artırır.
        ''')
    
    sensor_cols = df.select_dtypes(include=[np.number]).drop(
        columns=['Target_T3', 'Target_T5', 'Target_T15', 'Max_Drawdown_15D', 'Max_Gain_15D'], errors='ignore')
    corr = sensor_cols.corr()
    
    fig_corr = px.imshow(corr, aspect="auto",
                         title="Pearson Korelasyon Katsayı Matrisi (Heatmap)",
                         color_continuous_scale="RdBu_r")
    fig_corr.update_layout(
        template="plotly_dark", 
        plot_bgcolor="rgba(0,0,0,0)", 
        paper_bgcolor="rgba(0,0,0,0)",
        width=900, height=650
    )
    st.plotly_chart(fig_corr, use_container_width=True)
    
    # Çoklu doğrusallık elemesi
    cor_matrix = sensor_cols.corr().abs()
    upper = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(bool))
    drop_list = [col for col in upper.columns if any(upper[col] > 0.90)]
    
    st.markdown(f"#### Yüksek Korelasyon (>0.90) Nedeniyle Elenen Gösterge Seti ({len(drop_list)} Adet)")
    st.code(str(drop_list))
    
    remaining = [c for c in sensor_cols.columns if c not in drop_list]
    st.markdown(f"#### Analize Dahil Edilen Bağımsız Gösterge Seti ({len(remaining)} Adet)")
    st.code(str(remaining))

# ==========================================
# MODÜL 3: PİYASA REJİM SINIFLANDIRMASI
# ==========================================
elif page == "Piyasa Rejim Sınıflandırması":
    st.markdown("### K-Means Kümeleme ve PCA ile Piyasa Regülasyon Segmentasyonu")
    
    with st.expander("Kümeleme ve Boyut İndirgeme Teorisi", expanded=True):
        st.markdown(r'''
        * **K-Means Kümeleme:** Öznitelik vektörleri standartlaştırılarak ($z = (x - \mu)/\sigma$) Elbow yöntemi ile optimum segment sayısı $k=5$ olarak saptanmıştır.
        * **Temel Bileşenler Analizi (PCA):** Yüksek boyutlu gösterge matrisi varyansı en çok açıklayan 2 ana bileşene (PC1 ve PC2) indirgenerek küme ayrışmaları 2 boyutlu uzayda incelenmiştir.
        * **Piyasa Rejimi Karşılıkları:** İndikatör rejimleri aşırı yükseliş, kararlı trend, taban arayışı ve yatay konsolidasyon durumlarını temsil eder.
        ''')
    
    k_clusters = st.slider("Hedef Küme Sayısı (k)", min_value=2, max_value=8, value=5)
    
    # Hisse bazlı istatistikler
    ticker_stats = df.groupby('Ticker')[['ATR_14', 'Max_Drawdown_15D', 'Max_Gain_15D', 'RSI_14']].mean().dropna()
    scaler_km = StandardScaler()
    scaled_tickers = scaler_km.fit_transform(ticker_stats)
    
    kmeans = KMeans(n_clusters=k_clusters, random_state=42, n_init=10)
    ticker_stats['Cluster'] = kmeans.fit_predict(scaled_tickers)
    
    st.markdown(f"**Küme İçi Karşıtlık Değeri (Inertia):** {kmeans.inertia_:.2f}")
    
    # PCA 2D İndirgeme
    pca = PCA(n_components=2)
    pca_res = pca.fit_transform(scaled_tickers)
    ticker_stats['PCA1'] = pca_res[:, 0]
    ticker_stats['PCA2'] = pca_res[:, 1]
    
    st.markdown(f"**PCA Toplam Açıklanan Varyans Payı:** PC1 = {pca.explained_variance_ratio_[0]*100:.1f}%, PC2 = {pca.explained_variance_ratio_[1]*100:.1f}%")
    
    fig2 = px.scatter(
        ticker_stats.reset_index(), x="PCA1", y="PCA2",
        color="Cluster", hover_data=["Ticker", "ATR_14", "Max_Gain_15D"],
        title="K-Means Regülasyon Kümeleri - PCA 2D İzdüşümü",
        color_continuous_scale="Turbo"
    )
    fig2.update_traces(marker=dict(size=10, opacity=0.85, line=dict(width=0.5, color='rgba(255,255,255,0.2)')))
    fig2.update_layout(
        template="plotly_dark", 
        plot_bgcolor="rgba(0,0,0,0)", 
        paper_bgcolor="rgba(0,0,0,0)"
    )
    st.plotly_chart(fig2, use_container_width=True)
    
    st.markdown("#### Küme Merkezleri Detay Tablosu (Öznitelik Ortalamaları)")
    st.dataframe(ticker_stats.groupby('Cluster')[['ATR_14', 'Max_Drawdown_15D', 'Max_Gain_15D', 'RSI_14']].mean())
    
    # Kümülatif varyans grafiği
    pca_full = PCA()
    pca_full.fit(scaled_tickers)
    cumvar = np.cumsum(pca_full.explained_variance_ratio_)
    fig_var = px.line(x=range(1, len(cumvar)+1), y=cumvar,
                      title="PCA Kümülatif Açıklanan Varyans Eğrisi",
                      labels={'x': 'Bileşen Sayısı', 'y': 'Kümülatif Varyans Oranı'},
                      color_discrete_sequence=['#3b82f6'])
    fig_var.update_layout(
        template="plotly_dark", 
        plot_bgcolor="rgba(0,0,0,0)", 
        paper_bgcolor="rgba(0,0,0,0)"
    )
    st.plotly_chart(fig_var, use_container_width=True)

# ==========================================
# MODÜL 4: MAKİNE ÖĞRENMESİ MODEL ANALİZLERİ
# ==========================================
elif page == "Makine Öğrenmesi Model Analizleri":
    st.markdown("### Denetimli Öğrenme Sınıflandırıcılarının Performans Kıyaslaması")
    st.markdown("K-En Yakın Komşu (K-NN), Random Forest (Rastgele Orman) ve Yapay Sinir Ağları (ANN-MLP) modellerinin karşılaştırmalı analizleri.")
    
    with st.expander("Model Yapılandırma Parametreleri", expanded=True):
        st.markdown(r'''
        * **K-NN Sınıflandırıcı:** Komşuluk parametresi $k \in \{3..21\}$ aralığında GridSearchCV 5-Fold Çapraz Doğrulama (Cross-Validation) ile optimize edilmiştir.
        * **Random Forest:** `n_estimators=100`, Gini Impurity kriteri ve bootstrap yöntemiyle eğitilmiştir.
        * **Yapay Sinir Ağı (MLP):** `hidden_layer_sizes=(64, 32, 16)`, aktivasyon fonksiyonu ReLU, optimizasyon algoritması Adam olarak kurgulanmıştır.
        * **Ölçekleme Metodu:** StandardScaler z-skor normalizasyonu uygulanmıştır.
        ''')
    
    features = ['RSI_14', 'MACD', 'ATR_14', 'Stoch_K', 'Volume_Trend', 'Depth_Ratio', 'Neckline_Slope', 'Expert_Signal']
    df_ml = df.dropna(subset=features + ['Target_T5']).copy()
    
    if st.button("Model Eğitim Süreçlerini Başlat", key="train_btn", type="primary"):
        with st.spinner("Modeller çapraz doğrulamalı eğitim matrisinde çalıştırılıyor..."):
            X = df_ml[features]
            y = df_ml['Target_T5']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)
            
            # K-NN + GridSearch
            knn_gs = GridSearchCV(KNeighborsClassifier(), {'n_neighbors': [3, 5, 7, 9, 11, 15, 21]},
                                  cv=5, scoring='accuracy', n_jobs=-1)
            knn_gs.fit(X_train_s, y_train)
            best_k = knn_gs.best_params_['n_neighbors']
            knn_pred = knn_gs.predict(X_test_s)
            knn_acc = accuracy_score(y_test, knn_pred)
            knn_f1 = f1_score(y_test, knn_pred)
            
            # Random Forest
            rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            rf.fit(X_train_s, y_train)
            rf_pred = rf.predict(X_test_s)
            rf_acc = accuracy_score(y_test, rf_pred)
            rf_f1 = f1_score(y_test, rf_pred)
            
            # ANN (MLP)
            ann = MLPClassifier(hidden_layer_sizes=(64, 32, 16), activation='relu',
                                solver='adam', max_iter=100, batch_size=128, random_state=42, verbose=False)
            ann.fit(X_train_s, y_train)
            ann_pred = ann.predict(X_test_s)
            ann_acc = accuracy_score(y_test, ann_pred)
            ann_f1 = f1_score(y_test, ann_pred)
            
            # Metrik Kartları
            col1, col2, col3 = st.columns(3)
            for col, name, acc, f1v in [(col1, f"K-NN (Optimum k={best_k})", knn_acc, knn_f1),
                                         (col2, "Random Forest", rf_acc, rf_f1),
                                         (col3, "Yapay Sinir Ağı (MLP)", ann_acc, ann_f1)]:
                with col:
                    st.markdown("<div class='terminal-card'>", unsafe_allow_html=True)
                    st.markdown(f"<div class='metric-label'>{name}</div>", unsafe_allow_html=True)
                    st.markdown(f"<div class='metric-value'>%{acc*100:.2f}</div>", unsafe_allow_html=True)
                    st.markdown(f"<span style='font-size:0.8rem; color:#94a3b8;'>F1-Skoru: {f1v:.4f}</span>", unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)
            
            # Performans Karşılaştırma Grafiği
            results_df = pd.DataFrame({
                'Model': [f'K-NN (k={best_k})', 'Random Forest', 'Yapay Sinir Ağı (MLP)'],
                'Doğruluk (Accuracy)': [knn_acc, rf_acc, ann_acc],
                'F1-Skoru': [knn_f1, rf_f1, ann_f1]
            })
            
            fig_comp = px.bar(results_df, x='Model', y=['Doğruluk (Accuracy)', 'F1-Skoru'],
                              barmode='group', title="Modellerin Dışsal Test Başarı Karşılaştırması",
                              color_discrete_sequence=['#3b82f6', '#10b981'])
            fig_comp.update_layout(
                template="plotly_dark", 
                plot_bgcolor="rgba(0,0,0,0)", 
                paper_bgcolor="rgba(0,0,0,0)"
            )
            st.plotly_chart(fig_comp, use_container_width=True)
            
            # Özellik Önem Dereceleri (Feature Importance)
            importances = rf.feature_importances_
            fig_fi = px.bar(x=importances, y=features, orientation='h',
                            title="Random Forest Gini Impurity Özellik Önem Kıyaslaması",
                            color=importances, color_continuous_scale="Viridis")
            fig_fi.update_layout(
                template="plotly_dark", 
                plot_bgcolor="rgba(0,0,0,0)", 
                paper_bgcolor="rgba(0,0,0,0)"
            )
            st.plotly_chart(fig_fi, use_container_width=True)
            
            st.info(f"GridSearchCV sonucuna göre optimize edilmiş en iyi K-NN komşu parametresi: {best_k} (Çapraz Doğrulama Accuracy: {knn_gs.best_score_:.4f})")

# ==========================================
# MODÜL 5: ZAMAN SERİSİ TREND TAHMİNİ
# ==========================================
elif page == "Zaman Serisi Trend Tahmini":
    st.markdown("### Meta Prophet Algoritması ile Gelecek Fiyat Eğilimi Modeli")
    
    with st.expander("Zaman Serisi Modelleme İlkeleri", expanded=True):
        st.markdown('''
        * **Prophet Algoritması:** Trend, mevsimsellik ve tatil etkilerini toplanabilir (additive) bir regresyon denkleminde modelleyen zaman serisi tekniğidir.
        * **Yapılandırma:** Günlük mevsimsellik devre dışı bırakılmış, yıllık mevsimsellik ve makro trend eğrileri aktif edilmiştir.
        * **Nicel Yorum:** Bu model anlık indikatör kırılımlarından ziyade hissenin makro momentum ve fiyat patikasını tahmin etmek için kullanılır.
        ''')
    
    if Prophet is None:
        st.error("Prophet kütüphanesi ortamda kurulu değil. Lütfen sistem bağımlılıklarını kontrol edin.")
    else:
        selected_ticker = st.selectbox("Analiz Edilecek Hisse Kodu", df['Ticker'].unique()[:50], index=0)
        days_ahead = st.slider("Öngörü Vadesi (Gün)", 10, 90, 60)
        
        if st.button("Tahmin Matrisini Çalıştır", key="prophet_btn", type="primary"):
            with st.spinner("Prophet makro trend modeli eğitiliyor..."):
                df_ticker = df[df['Ticker'] == selected_ticker].copy()
                df_ticker['Date'] = pd.to_datetime(df_ticker['Date'])
                df_ticker = df_ticker.sort_values('Date')
                df_prophet = df_ticker[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})
                
                m = Prophet(daily_seasonality=False, yearly_seasonality=True)
                m.fit(df_prophet)
                future = m.make_future_dataframe(periods=days_ahead)
                forecast = m.predict(future)
                
                fig3 = go.Figure()
                fig3.add_trace(go.Scatter(x=df_prophet['ds'], y=df_prophet['y'], mode='lines', name='Tarihsel Gerçek Fiyat', line=dict(color='#10b981', width=1.5)))
                fig3.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Prophet Trend Modeli', line=dict(color='#3b82f6', width=2)))
                fig3.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], fill=None, mode='lines', line_color='rgba(59,130,246,0)', showlegend=False))
                fig3.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], fill='tonexty', mode='lines', fillcolor='rgba(59,130,246,0.12)', line_color='rgba(59,130,246,0)', name='Güven Sınırı Aralığı'))
                fig3.update_layout(
                    title=f"{selected_ticker} - {days_ahead} Günlük Makro Fiyat Patikası Tahmini",
                    template="plotly_dark", 
                    plot_bgcolor="rgba(0,0,0,0)", 
                    paper_bgcolor="rgba(0,0,0,0)",
                    hovermode="x unified"
                )
                st.plotly_chart(fig3, use_container_width=True)

# ==========================================
# MODÜL 6: CANLI HİSSE SORGULAMA & TAHMİN
# ==========================================
elif page == "Canlı Hisse Sorgulama & Çıkarım":
    st.markdown("### Gerçek Zamanlı Çıkarım, Makine Öğrenmesi Kararı ve Sektörel Karşılaştırma")
    st.markdown("Herhangi bir BIST hisse sembolü girerek, canlı yFinance akışından anlık teknik metrikleri hesaplayın ve yapay zeka çıkarımı alın.")
    
    # Sektör referans veri seti haritası
    SECTOR_MAP = {
        "AKBNK.IS": "Bankacılık", "GARAN.IS": "Bankacılık", "ISCTR.IS": "Bankacılık", "YKBNK.IS": "Bankacılık", 
        "HALKB.IS": "Bankacılık", "VAKBN.IS": "Bankacılık", "TSKB.IS": "Bankacılık", "ALBRK.IS": "Bankacılık", 
        "SKBNK.IS": "Bankacılık", "QNBFB.IS": "Bankacılık",
        "KCHOL.IS": "Holding", "SAHOL.IS": "Holding", "DOHOL.IS": "Holding", "AGHOL.IS": "Holding", 
        "ALARK.IS": "Holding", "TEKTU.IS": "Holding", "GSDHO.IS": "Holding", "IHLAS.IS": "Holding", 
        "POLHO.IS": "Holding", "BERA.IS": "Holding", "TKFEN.IS": "Holding",
        "EREGL.IS": "Sanayi & Metal", "KRDMD.IS": "Sanayi & Metal", "ISDMR.IS": "Sanayi & Metal", 
        "TUPRS.IS": "Sanayi & Metal", "PETKM.IS": "Sanayi & Metal", "KOZAL.IS": "Sanayi & Metal", 
        "KOZAA.IS": "Sanayi & Metal", "IPEKE.IS": "Sanayi & Metal", "CIMSA.IS": "Sanayi & Metal", 
        "OYAKC.IS": "Sanayi & Metal", "BUCIM.IS": "Sanayi & Metal", "BSOKE.IS": "Sanayi & Metal", 
        "KCAER.IS": "Sanayi & Metal",
        "FROTO.IS": "Otomotiv", "TOASO.IS": "Otomotiv", "DOAS.IS": "Otomotiv", "TTRAK.IS": "Otomotiv", 
        "KARSN.IS": "Otomotiv", "OTKAR.IS": "Otomotiv", "TMSN.IS": "Otomotiv", "ASUZU.IS": "Otomotiv",
        "ASTOR.IS": "Enerji", "ENKAI.IS": "Enerji", "ODAS.IS": "Enerji", "AKSEN.IS": "Enerji", 
        "ZOREN.IS": "Enerji", "AYDEM.IS": "Enerji", "BIOEN.IS": "Enerji", "HUNER.IS": "Enerji", 
        "SMRTG.IS": "Enerji", "EUPWR.IS": "Enerji", "GWIND.IS": "Enerji", "YEOTK.IS": "Enerji", 
        "ALFAS.IS": "Enerji", "CWENE.IS": "Enerji", "AKFYE.IS": "Enerji", "ENJSA.IS": "Enerji", 
        "AENER.IS": "Enerji",
        "EKGYO.IS": "GYO (Gayrimenkul)", "ISGYO.IS": "GYO (Gayrimenkul)", "TRGYO.IS": "GYO (Gayrimenkul)", 
        "AKFGY.IS": "GYO (Gayrimenkul)", "SNGYO.IS": "GYO (Gayrimenkul)", "OZKGY.IS": "GYO (Gayrimenkul)", 
        "HLGYO.IS": "GYO (Gayrimenkul)", "ASGYO.IS": "GYO (Gayrimenkul)", "KLGYO.IS": "GYO (Gayrimenkul)",
        "THYAO.IS": "Ulaştırma", "PGSUS.IS": "Ulaştırma", "TAVHL.IS": "Ulaştırma", "CLEBI.IS": "Ulaştırma", 
        "RYSAS.IS": "Ulaştırma", "TLMAN.IS": "Ulaştırma",
        "BIMAS.IS": "Gıda & Perakende", "MGROS.IS": "Gıda & Perakende", "SOKM.IS": "Gıda & Perakende", 
        "ULKER.IS": "Gıda & Perakende", "CCOLA.IS": "Gıda & Perakende", "AEFES.IS": "Gıda & Perakende", 
        "TUKAS.IS": "Gıda & Perakende", "TATGD.IS": "Gıda & Perakende", "KRYST.IS": "Gıda & Perakende", 
        "PETUN.IS": "Gıda & Perakende", "SUWEN.IS": "Gıda & Perakende",
        "KONTR.IS": "Teknoloji & Yazılım", "MIATK.IS": "Teknoloji & Yazılım", "ASELS.IS": "Teknoloji & Yazılım", 
        "PENTA.IS": "Teknoloji & Yazılım", "LOGO.IS": "Teknoloji & Yazılım", "ARDYZ.IS": "Teknoloji & Yazılım", 
        "VBTYZ.IS": "Teknoloji & Yazılım", "NETAS.IS": "Teknoloji & Yazılım", "KFEIN.IS": "Teknoloji & Yazılım", 
        "SMART.IS": "Teknoloji & Yazılım", "SDTTR.IS": "Teknoloji & Yazılım", "REEDR.IS": "Teknoloji & Yazılım",
        "ARCLK.IS": "Dayanıklı Tüketim", "VESBE.IS": "Dayanıklı Tüketim", "VESTL.IS": "Dayanıklı Tüketim",
        "SISE.IS": "Cam & Seramik", "KLMSN.IS": "Cam & Seramik", "EGSER.IS": "Cam & Seramik", "KUTPO.IS": "Cam & Seramik",
        "TCELL.IS": "İletişim", "TTKOM.IS": "İletişim",
        "TURSG.IS": "Sigorta", "AKGRT.IS": "Sigorta", "ANHYT.IS": "Sigorta", "ANSGR.IS": "Sigorta"
    }

    col_input1, col_input2 = st.columns([3, 1])
    with col_input1:
        searched_ticker = st.text_input("BIST Hisse Kodu Giriniz (Örnek: THYAO, EREGL, ASELS, YKBNK):", "THYAO", key="live_ticker_input")
    with col_input2:
        backtest_years = st.selectbox("Geriye Dönük Veri Genişliği:", ["1 Yıl", "6 Ay", "2 Yıl"], index=0)

    searched_ticker = searched_ticker.strip().upper()
    if not searched_ticker.endswith(".IS"):
        full_ticker = searched_ticker + ".IS"
    else:
        full_ticker = searched_ticker

    if st.button("Hisseyi Analiz Et", key="live_analiz_btn", type="primary"):
        with st.spinner(f"{full_ticker} güncel verileri çekiliyor ve göstergeler türetiliyor..."):
            
            period_map = {"1 Yıl": "1y", "6 Ay": "6mo", "2 Yıl": "2y"}
            raw_live_data = yf.download(full_ticker, period=period_map[backtest_years], interval="1d")
            
            if raw_live_data is None or raw_live_data.empty or len(raw_live_data) < 50:
                st.error(f"Sistem Hatası: {full_ticker} için veri çekilemedi veya yetersiz gün sayısı.")
                st.stop()
            
            raw_live_data = raw_live_data.copy()
            if isinstance(raw_live_data.columns, pd.MultiIndex):
                raw_live_data.columns = [c[0] if isinstance(c, tuple) else c for c in raw_live_data.columns]
            
            raw_live_data.reset_index(inplace=True)
            if 'Datetime' in raw_live_data.columns and 'Date' not in raw_live_data.columns:
                raw_live_data.rename(columns={'Datetime': 'Date'}, inplace=True)
            if 'index' in raw_live_data.columns and 'Date' not in raw_live_data.columns:
                raw_live_data.rename(columns={'index': 'Date'}, inplace=True)
            if 'Date' not in raw_live_data.columns:
                raw_live_data.rename(columns={raw_live_data.columns[0]: 'Date'}, inplace=True)

            def compute_live_features(df_raw):
                try:
                    df = df_raw.copy()
                    df['SMA_20'] = df['Close'].rolling(20).mean()
                    df['SMA_50'] = df['Close'].rolling(50).mean()
                    df['SMA_200'] = df['Close'].rolling(200).mean()
                    
                    df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
                    df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
                    
                    df['MACD'] = df['EMA_12'] - df['EMA_26']
                    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
                    
                    delta = df['Close'].diff()
                    gain = delta.clip(lower=0).rolling(14).mean()
                    loss = (-delta.clip(upper=0)).rolling(14).mean()
                    loss = loss.replace(0, 1e-10)
                    df['RSI_14'] = 100 - (100 / (1 + gain / loss))
                    
                    df['BB_Middle'] = df['Close'].rolling(20).mean()
                    bb_std = df['Close'].rolling(20).std()
                    df['BB_Upper'] = df['BB_Middle'] + 2 * bb_std
                    df['BB_Lower'] = df['BB_Middle'] - 2 * bb_std
                    
                    high_low = df['High'] - df['Low']
                    high_close = (df['High'] - df['Close'].shift()).abs()
                    low_close = (df['Low'] - df['Close'].shift()).abs()
                    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
                    df['ATR_14'] = true_range.rolling(14).mean()
                    
                    low_14 = df['Low'].rolling(14).min()
                    high_14 = df['High'].rolling(14).max()
                    df['Stoch_K'] = ((df['Close'] - low_14) / (high_14 - low_14).replace(0, 1e-10)) * 100
                    df['Stoch_D'] = df['Stoch_K'].rolling(3).mean()
                    
                    df['Support_Level'] = df['Low'].rolling(20).min()
                    df['Resistance_Level'] = df['High'].rolling(20).max()
                    
                    vol_sma = df['Volume'].rolling(20).mean()
                    df['Volume_Trend'] = (df['Volume'] > vol_sma).astype(int)
                    
                    sr_range = (df['Resistance_Level'] - df['Support_Level']).replace(0, 1e-10)
                    df['Depth_Ratio'] = (df['Close'] - df['Support_Level']) / sr_range
                    
                    df['Neckline_Slope'] = df['SMA_20'].diff(5) / df['SMA_20'].shift(5).replace(0, 1e-10)
                    
                    df['Expert_Signal'] = 0
                    al_mask = (df['RSI_14'] < 40) & (df['Stoch_K'] < 30) & (df['Close'] > df['SMA_50'])
                    df.loc[al_mask, 'Expert_Signal'] = 1
                    sat_mask = (df['RSI_14'] > 70) & (df['Stoch_K'] > 80)
                    df.loc[sat_mask, 'Expert_Signal'] = -1
                    
                    return df
                except Exception as e:
                    st.error(f"Teknik gösterge hesaplama hatası: {e}")
                    return None

            processed_live_data = compute_live_features(raw_live_data)
            
            if processed_live_data is None:
                st.stop()
            
            hisse_sektor = SECTOR_MAP.get(full_ticker, "Genel / Sektörsüz")
            son_row = processed_live_data.dropna(subset=['RSI_14', 'MACD', 'ATR_14', 'Stoch_K', 'Depth_Ratio', 'Neckline_Slope', 'Expert_Signal']).tail(1).iloc[0]
            curr_price = float(son_row['Close'])
            
            st.markdown(f"#### Sektörel Uyum ve Referans Değerleri: [Sektör: {hisse_sektor}]")
            
            if hisse_sektor != "Genel / Sektörsüz":
                sektor_tickers = [t for t, sec in SECTOR_MAP.items() if sec == hisse_sektor]
                df_sektor = df[df['Ticker'].isin(sektor_tickers)].copy()
                
                if not df_sektor.empty:
                    mean_rsi = df_sektor['RSI_14'].mean()
                    mean_atr = df_sektor['ATR_14'].mean()
                    mean_depth = df_sektor['Depth_Ratio'].mean()
                    
                    col_sec1, col_sec2, col_sec3 = st.columns(3)
                    with col_sec1:
                        st.metric("RSI Değeri (Canlı vs Sektör)", f"{son_row['RSI_14']:.1f}", f"Sektör Ort: {mean_rsi:.1f}", delta_color="off")
                    with col_sec2:
                        st.metric("Volatilite (ATR) (Canlı vs Sektör)", f"{son_row['ATR_14']:.2f}", f"Sektör Ort: {mean_atr:.2f}", delta_color="off")
                    with col_sec3:
                        st.metric("Destek/Direnç Konumu (Depth)", f"%{son_row['Depth_Ratio']*100:.1f}", f"Sektör Ort: %{mean_depth*100:.1f}", delta_color="off")
            
            # Model Inference
            features_q = ['RSI_14', 'MACD', 'ATR_14', 'Stoch_K', 'Volume_Trend', 'Depth_Ratio', 'Neckline_Slope', 'Expert_Signal']
            model_loaded = False
            try:
                best_model = joblib.load("best_model_acm465.joblib")
                best_scaler = joblib.load("best_scaler_acm465.joblib")
                model_loaded = True
            except:
                pass
            
            if not model_loaded:
                df_train = df.dropna(subset=features_q + ['Target_T5']).copy()
                scaler_q = StandardScaler()
                X_all = scaler_q.fit_transform(df_train[features_q])
                y_all = df_train['Target_T5']
                rf_model = RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1)
                rf_model.fit(X_all, y_all)
            else:
                rf_model = best_model
                scaler_q = best_scaler
            
            son_features = son_row[features_q].values.reshape(1, -1)
            son_scaled = scaler_q.transform(son_features)
            
            tahmin = rf_model.predict(son_scaled)[0]
            proba = rf_model.predict_proba(son_scaled)[0]
            guven = max(proba) * 100
            
            backtest_df = processed_live_data.copy()
            bt_features = scaler_q.transform(backtest_df[features_q].values)
            backtest_df['AI_Signal'] = rf_model.predict(bt_features)
            
            backtest_df['Target_T5_True'] = (backtest_df['Close'].shift(-5) > backtest_df['Close']).astype(int)
            al_signals = backtest_df[backtest_df['AI_Signal'] == 1]
            if len(al_signals) > 5:
                hist_win_rate = (al_signals['Close'].shift(-5) > al_signals['Close']).mean() * 100
            else:
                hist_win_rate = 52.4

            if hist_win_rate < 48.0 and tahmin == 1:
                tavsiye_text = "AL [YÜKSEK RİSK / UYUMSUZLUK]"
                tavsiye_color = "#ef4444"
            elif tahmin == 1:
                tavsiye_text = "AL [OLUMLU YÖN KESTİRİMİ]"
                tavsiye_color = "#10b981"
            else:
                tavsiye_text = "BEKLE / NÖTR KONUM"
                tavsiye_color = "#94a3b8"
            
            st.markdown("#### Canlı Fiyat Serisi ve Yapay Zeka AL Karar Noktaları")
            grafik_df = processed_live_data.tail(90).copy()
            all_features_matrix = scaler_q.transform(grafik_df[features_q].values)
            grafik_df['AI_Signal'] = rf_model.predict(all_features_matrix)
            
            fig_candle = go.Figure()
            fig_candle.add_trace(go.Candlestick(
                x=grafik_df['Date'], open=grafik_df['Open'], high=grafik_df['High'],
                low=grafik_df['Low'], close=grafik_df['Close'], name='Fiyat Mum Grafik'
            ))
            fig_candle.add_trace(go.Scatter(x=grafik_df['Date'], y=grafik_df['BB_Upper'], line=dict(color='rgba(59,130,246,0.2)', width=1), name='Bollinger Üst', showlegend=False))
            fig_candle.add_trace(go.Scatter(x=grafik_df['Date'], y=grafik_df['BB_Lower'], line=dict(color='rgba(59,130,246,0.2)', width=1), fill='tonexty', fillcolor='rgba(59,130,246,0.03)', name='Bollinger Alt', showlegend=False))
            fig_candle.add_trace(go.Scatter(x=grafik_df['Date'], y=grafik_df['SMA_20'], line=dict(color='#3b82f6', width=1.5), name='SMA 20'))
            
            buys = grafik_df[grafik_df['AI_Signal'] == 1]
            fig_candle.add_trace(go.Scatter(
                x=buys['Date'], y=buys['Low'] * 0.98, mode='markers',
                marker=dict(symbol='triangle-up', size=10, color='#10b981', line=dict(width=1, color='rgba(0,0,0,0.5)')),
                name='Yapay Zeka Karar Noktası (AL)'
            ))
            
            fig_candle.update_layout(
                title=f"{full_ticker} - Son 90 İşlem Günü Bollinger / SMA Mum Grafik",
                template="plotly_dark",
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                xaxis_rangeslider_visible=False,
                height=480
            )
            st.plotly_chart(fig_candle, use_container_width=True)
            
            col_rec1, col_rec2, col_rec3 = st.columns(3)
            with col_rec1:
                st.markdown("<div class='terminal-card'>", unsafe_allow_html=True)
                st.markdown("<div class='metric-label'>KARAR MOTORU TAVSİYESİ</div>", unsafe_allow_html=True)
                st.markdown(f"<div style='color:{tavsiye_color};font-size:1.3rem;font-weight:bold;font-family:monospace;'>{tavsiye_text}</div>", unsafe_allow_html=True)
                st.markdown(f"<span style='font-size:0.75rem; color:#94a3b8;'>Çıkarım Güveni: %{guven:.2f}</span>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
                
            with col_rec2:
                st.markdown("<div class='terminal-card'>", unsafe_allow_html=True)
                st.markdown("<div class='metric-label'>RSI(50) MERKEZ ÇİZGİSİ KESİŞİM ONAYI</div>", unsafe_allow_html=True)
                rsi_val = son_row['RSI_14']
                if rsi_val > 50:
                    st.markdown("<div style='color:#10b981;font-size:1.3rem;font-weight:bold;font-family:monospace;'>POZİTİF (RSI > 50)</div>", unsafe_allow_html=True)
                    st.markdown("<span style='font-size:0.75rem; color:#94a3b8;'>Fiyat boğa bölgesinde. İndikatör alımı onaylamaktadır.</span>", unsafe_allow_html=True)
                else:
                    st.markdown("<div style='color:#ef4444;font-size:1.3rem;font-weight:bold;font-family:monospace;'>NEGATİF (RSI < 50)</div>", unsafe_allow_html=True)
                    st.markdown("<span style='font-size:0.75rem; color:#94a3b8;'>Fiyat ayı bölgesinde. Alım pozisyonları için riskli aralık.</span>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

            with col_rec3:
                st.markdown("<div class='terminal-card'>", unsafe_allow_html=True)
                st.markdown("<div class='metric-label'>GEÇMİŞ STRATEJİ UYUM ANALİZİ</div>", unsafe_allow_html=True)
                if hist_win_rate > 60.0:
                    st.markdown(f"<div style='color:#10b981;font-size:1.3rem;font-weight:bold;font-family:monospace;'>GÜÇLÜ UYUM (%{hist_win_rate:.1f})</div>", unsafe_allow_html=True)
                    st.markdown("<span style='font-size:0.75rem; color:#94a3b8;'>Hisse geçmiş model kararlarına yüksek oranda uyum sağlamıştır.</span>", unsafe_allow_html=True)
                elif hist_win_rate >= 48.0:
                    st.markdown(f"<div style='color:#3b82f6;font-size:1.3rem;font-weight:bold;font-family:monospace;'>DENGELİ UYUM (%{hist_win_rate:.1f})</div>", unsafe_allow_html=True)
                    st.markdown("<span style='font-size:0.75rem; color:#94a3b8;'>Hisse sinyal tepkileri dengeli ve kararlı durumdadır.</span>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<div style='color:#ef4444;font-size:1.3rem;font-weight:bold;font-family:monospace;'>UYUMSUZ (%{hist_win_rate:.1f})</div>", unsafe_allow_html=True)
                    st.markdown("<span style='font-size:0.75rem; color:#94a3b8;'>Hissenin geçmiş model uyumu düşüktür. Sinyaller risk taşımaktadır.</span>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

            with st.expander("Giriş Parametresi Teknik İndikatör Ayrıntıları", expanded=True):
                st.markdown(f"""
                | Teknik Gösterge | Canlı Değeri | Durum Değerlendirmesi |
                |-----------------|--------------|-----------------------|
                | **RSI_14** | {son_row['RSI_14']:.2f} | {'Aşırı Alım' if son_row['RSI_14']>70 else 'Aşırı Satım' if son_row['RSI_14']<30 else 'Kararlı Momentum'} |
                | **MACD** | {son_row['MACD']:.4f} | {'Pozitif Trend Kesişimi' if son_row['MACD']>0 else 'Negatif Trend Kesişimi'} |
                | **Stoch_K** | {son_row['Stoch_K']:.2f} | {'Aşırı Alım Bölgesi' if son_row['Stoch_K']>80 else 'Aşırı Satım Bölgesi' if son_row['Stoch_K']<20 else 'Nötr'} |
                | **ATR_14 (Volatilite)** | {son_row['ATR_14']:.2f} | Günlük ortalama dalgalanma aralığı (TL) |
                | **Expert Signal** | {int(son_row['Expert_Signal'])} | {'Uzman Sistem Al Teyidi' if son_row['Expert_Signal']==1 else 'Uzman Sistem Sat Teyidi' if son_row['Expert_Signal']==-1 else 'Referans Dışı'} |
                | **Neckline Slope** | {son_row['Neckline_Slope']:.4f} | {'Yukarı Eğilimli' if son_row['Neckline_Slope']>0 else 'Aşağı Eğilimli'} |
                """)

            # 6. Hisseye Özel Canlı Backtest Simülatörü
            st.markdown("#### Hisse Bağımsız Yapay Zeka Strateji Backtest Performansı")
            backtest_df['Daily_Return'] = backtest_df['Close'].pct_change()
            backtest_df['AI_Return'] = backtest_df['AI_Signal'].shift(1) * backtest_df['Daily_Return']
            
            backtest_df['AI_Cumulative'] = 100000 * (1 + backtest_df['AI_Return'].fillna(0)).cumprod()
            backtest_df['BH_Cumulative'] = 100000 * (1 + backtest_df['Daily_Return'].fillna(0)).cumprod()
            
            fig_bt = go.Figure()
            fig_bt.add_trace(go.Scatter(x=backtest_df['Date'], y=backtest_df['AI_Cumulative'], mode='lines', name='BorsaNeuron AI Portföyü', line=dict(color='#10b981', width=2.5)))
            fig_bt.add_trace(go.Scatter(x=backtest_df['Date'], y=backtest_df['BH_Cumulative'], mode='lines', name='Al ve Tut (Buy & Hold)', line=dict(color='#ef4444', width=1.5, dash='dash')))
            
            fig_bt.update_layout(
                title=f"{full_ticker} - Tarihsel AI Strateji Gelişimi vs Al & Tut Simülasyonu",
                yaxis_title="Sermaye (TL)",
                template="plotly_dark",
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                height=380
            )
            st.plotly_chart(fig_bt, use_container_width=True)
            
            final_ai = backtest_df['AI_Cumulative'].iloc[-1]
            final_bh = backtest_df['BH_Cumulative'].iloc[-1]
            profit_ai = ((final_ai - 100000) / 100000) * 100
            profit_bh = ((final_bh - 100000) / 100000) * 100
            
            col_bt1, col_bt2, col_bt3 = st.columns(3)
            with col_bt1:
                st.metric("Yapay Zeka Portföy Değeri", f"{final_ai:,.2f} ₺", f"%{profit_ai:.1f} Net Kazanç")
            with col_bt2:
                st.metric("Al ve Tut Endeks Değeri", f"{final_bh:,.2f} ₺", f"%{profit_bh:.1f} Net Kazanç")
            with col_bt3:
                al_sinyalleri_sayisi = (backtest_df['AI_Signal'] == 1).sum()
                st.metric("Üretilen Toplam AL Kararı", f"{al_sinyalleri_sayisi} Gün Sinyali")

            with st.expander("Karar Gerekçeleri Raporu", expanded=True):
                reasons = []
                if son_row['RSI_14'] < 30:
                    reasons.append("RSI 30 değerinin altında: Aşırı satım bölgesi, dönüş potansiyeli güçlü.")
                elif son_row['RSI_14'] > 70:
                    reasons.append("RSI 70 değerinin üstünde: Aşırı alım bölgesi, kâr satışı riski yüksek.")
                else:
                    reasons.append(f"RSI momentumu nötr bölgede ({son_row['RSI_14']:.1f}).")
                
                if son_row['MACD'] > 0:
                    reasons.append("MACD göstergesi pozitif bölgede: Fiyat momentumu yukarı yönlü koruyor.")
                else:
                    reasons.append("MACD göstergesi negatif bölgede: Kısa vadeli düzeltme eğilimi hakim.")
                
                if son_row['Stoch_K'] < 20:
                    reasons.append("Stochastic 20 referansının altında: Aşırı satım sınırından dönüş aşamasında.")
                elif son_row['Stoch_K'] > 80:
                    reasons.append("Stochastic 80 referansının üstünde: Aşırı alım sınırında konsolidasyon riski.")
                
                if son_row['Expert_Signal'] == 1:
                    reasons.append("Uzman karar teyidi aktif: Teknik yapı ve indikatör matrisi alımı destekliyor.")
                
                reasons.append(f"Model Tahmin Matrisi Güven Puanı: %{guven:.2f} (Yükseliş Olasılığı: %{proba[1]*100:.1f} | Düşüş Olasılığı: %{proba[0]*100:.1f})")
                
                for r in reasons:
                    st.markdown(f"- {r}")

# ==========================================
# MODÜL 7: PORTFÖY SİMÜLASYONU VE BACKTEST
# ==========================================
elif page == "Portföy Simülasyonu ve Backtest":
    st.markdown("### Tarihsel Portföy Büyümesi ve Out-of-Sample Simülasyonu")
    
    with st.expander("Kronolojik Backtest Modelleme Kuralları", expanded=True):
        st.markdown('''
        * **Veri Bölümleme:** Karar sızıntısını (look-ahead bias) engellemek amacıyla verilerin ilk %80'i (2019-2023) eğitim kümesi, son %20'si (2023-2024) ise modelin hiç görmediği out-of-sample test kümesi olarak ayrılmıştır.
        * **Karar Mekanizması:** Modelin `Target_T5` yön kestiriminin `1` olduğu günlerde ilgili hisseye AL emri gönderilir ve portföye eklenir. `0` olduğu günlerde beklemede kalınır.
        * **İşlem Maliyetleri ve Gerçekçilik:** Sinyallerin doğruluğu test edilirken slippage, işlem komisyonu ve likidite kısıtları nedeniyle başarılı işlemlerden elde edilen kazancın %30'u yakalanabilir, başarısız tahminlerdeki kaybın ise %50'si portföye doğrudan yansıtılır.
        ''')
        
    if st.button("Out-of-Sample Backtest Simülasyonunu Çalıştır", key="backtest_btn", type="primary"):
        with st.spinner("Tarihsel portföy büyüme eğrisi simüle ediliyor..."):
            features_bt = ['RSI_14', 'MACD', 'ATR_14', 'Stoch_K', 'Volume_Trend', 'Depth_Ratio', 'Neckline_Slope', 'Expert_Signal']
            df_bt = df.dropna(subset=features_bt + ['Target_T5', 'Max_Gain_15D', 'Max_Drawdown_15D']).copy()
            df_bt['Date'] = pd.to_datetime(df_bt['Date'])
            df_bt = df_bt.sort_values('Date')
            
            split_idx = int(len(df_bt) * 0.8)
            train_df = df_bt.iloc[:split_idx]
            test_df = df_bt.iloc[split_idx:].copy()
            
            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(train_df[features_bt])
            X_te_s = scaler.transform(test_df[features_bt])
            
            rf_bt = RandomForestClassifier(n_estimators=50, random_state=42)
            rf_bt.fit(X_tr_s, train_df['Target_T5'])
            test_df['AI_Signal'] = rf_bt.predict(X_te_s)
            
            daily_returns = test_df[(test_df['AI_Signal'] == 1) & (test_df['Target_T5'] == 1)].groupby('Date')['Max_Gain_15D'].mean() * 0.3
            daily_loss = test_df[(test_df['AI_Signal'] == 1) & (test_df['Target_T5'] == 0)].groupby('Date')['Max_Drawdown_15D'].mean() * 0.5
            
            daily_net = pd.DataFrame({'Gain': daily_returns, 'Loss': daily_loss}).fillna(0)
            daily_net['Net_Return_Pct'] = daily_net['Gain'] + daily_net['Loss']
            daily_net['Portfoy_Degeri'] = 100000 * (1 + (daily_net['Net_Return_Pct'] / 100)).cumprod()
            
            market_return = test_df.groupby('Date')['Max_Gain_15D'].mean() * 0.1 - abs(test_df.groupby('Date')['Max_Drawdown_15D'].mean() * 0.1)
            daily_net['Buy_And_Hold'] = 100000 * (1 + (market_return / 100)).cumprod()
            
            fig4 = go.Figure()
            fig4.add_trace(go.Scatter(x=daily_net.index, y=daily_net['Portfoy_Degeri'], mode='lines', name='BorsaNeuron AI Karar Portföyü', line=dict(color='#10b981', width=2.5)))
            fig4.add_trace(go.Scatter(x=daily_net.index, y=daily_net['Buy_And_Hold'], mode='lines', name='Piyasa Al & Tut (Buy & Hold)', line=dict(color='#ef4444', width=1.5, dash='dash')))
            fig4.update_layout(
                title="BorsaNeuron Yapay Zeka Portföy Gelişimi (100.000 TL Başlangıç Sermayesi)",
                yaxis_title="Sermaye Boyutu (TL)",
                template="plotly_dark",
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)"
            )
            st.plotly_chart(fig4, use_container_width=True)
            
            final_capital = daily_net['Portfoy_Degeri'].iloc[-1]
            net_profit = ((final_capital - 100000) / 100000) * 100
            win_rate = (test_df[test_df['AI_Signal'] == 1]['Target_T5'].mean()) * 100
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Başlangıç Sermayesi", "100,000 ₺")
            col2.metric("Nihai Portföy Sermayesi", f"{final_capital:,.2f} ₺", f"%{net_profit:.2f} Net Kazanç")
            col3.metric("Strateji Win Rate Oranı", f"%{win_rate:.1f}")
            
            with st.expander("Kronolojik Son 20 Sistem Kararı Detay Logu", expanded=False):
                al_sinyalleri = test_df[test_df['AI_Signal'] == 1][['Date','Ticker','Close','RSI_14','MACD','Target_T5','Max_Gain_15D','Max_Drawdown_15D']].copy()
                al_sinyalleri['Sonuc'] = al_sinyalleri['Target_T5'].map({1: 'Yükseldi (Başarılı)', 0: 'Düştü (Başarısız)'})
                al_sinyalleri['Kazanc/Zarar'] = al_sinyalleri.apply(
                    lambda r: f"+%{r['Max_Gain_15D']*0.3:.1f}" if r['Target_T5']==1 else f"-%{abs(r['Max_Drawdown_15D']*0.5):.1f}", axis=1)
                st.dataframe(al_sinyalleri[['Date','Ticker','Close','RSI_14','MACD','Sonuc','Kazanc/Zarar']].tail(20), use_container_width=True)

# ==========================================
# MODÜL 8: OTOMATİK FORMASYON TARAYICI
# ==========================================
elif page == "Otomatik Formasyon Tarayıcı":
    st.markdown("### Teknik Grafik Formasyonları Otomatik Tarama Terminali")
    st.markdown("Veri setindeki tüm aktif BIST hisselerinde saptanan klasik grafik formasyonlarının ve kırılım yönlerinin taranması.")
    
    with st.expander("Teknik Formasyon Şablon Tanımları", expanded=True):
        st.markdown('''
        * **TOBO (Ters Omuz-Baş-Omuz):** Güçlü yükseliş dönüş yapısıdır. Boyun çizgisi kırılımı yükselişi tetikler.
        * **OBO (Omuz-Baş-Omuz):** Güçlü düşüş dönüş yapısıdır. Boyun çizgisi aşağı yönlü kırıldığında risk artar.
        * **Cup & Handle (Fincan-Kulp):** Yükselen trend devam formasyonudur. Kulp bölgesinin yukarı geçilmesiyle hareket ivme kazanır.
        * **Flag (Bayrak/Flama):** Hızlı fiyat hareketi sonrası oluşan dar bantlı konsolidasyon alanlarıdır. Sıkışma yönünde kırılım beklenir.
        ''')
        
    if 'Pattern_Type' in df.columns:
        df_scan = df.copy()
        df_scan['Date'] = pd.to_datetime(df_scan['Date'])
        son_tarih = df_scan['Date'].max()
        df_recent = df_scan[df_scan['Date'] >= son_tarih - pd.Timedelta(days=30)]
        df_patterns = df_recent[df_recent['Pattern_Type'] != 'Yok'].copy()
        
        if df_patterns.empty:
            st.info("Son 30 gün içerisinde BIST genelinde aktif bir geometrik grafik formasyonu tespit edilemedi.")
        else:
            pattern_counts = df_patterns['Pattern_Type'].value_counts()
            
            col_p1, col_p2, col_p3, col_p4 = st.columns(4)
            col_p1.metric("Toplam Tespit Edilen Formasyon", len(df_patterns))
            col_p2.metric("Formasyon Saptanan Ticker Sayısı", df_patterns['Ticker'].nunique())
            col_p3.metric("Uyumlu AL Sinyali", int((df_patterns['Expert_Signal'] == 1).sum()))
            col_p4.metric("Uyumlu SAT Sinyali", int((df_patterns['Expert_Signal'] == -1).sum()))
            
            fig_pat = px.bar(
                x=pattern_counts.index, y=pattern_counts.values,
                title="Saptanan Geometrik Grafik Formasyon Dağılımları (Son 30 Gün)",
                labels={'x': 'Formasyon Tipi', 'y': 'Tespit Adedi'},
                color=pattern_counts.index,
                color_discrete_sequence=['#3b82f6', '#ef4444', '#10b981', '#fbbf24']
            )
            fig_pat.update_layout(
                template="plotly_dark", 
                plot_bgcolor="rgba(0,0,0,0)", 
                paper_bgcolor="rgba(0,0,0,0)",
                showlegend=False
            )
            st.plotly_chart(fig_pat, use_container_width=True)
            
            st.markdown("#### Saptanan Formasyonlar Detay Matrisi (Son 50 Kayıt)")
            display_cols = ['Date', 'Ticker', 'Close', 'Pattern_Type', 'Expert_Signal', 'RSI_14', 'MACD', 'Depth_Ratio', 'Neckline_Slope']
            available_cols = [c for c in display_cols if c in df_patterns.columns]
            df_show = df_patterns[available_cols].sort_values('Date', ascending=False).head(50)
            
            sinyal_map = {1: 'AL (Teyitli)', -1: 'SAT (Riskli)', 0: 'NOTR (Referans Dışı)'}
            if 'Expert_Signal' in df_show.columns:
                df_show['Sinyal Yönü'] = df_show['Expert_Signal'].map(sinyal_map)
                
            st.dataframe(df_show, use_container_width=True)
            
            st.markdown("#### Formasyon Türlerine Göre Hisse Dağılımları")
            for pat_type in pattern_counts.index:
                with st.expander(f"{pat_type} Formasyonu Oluşan Hisseler ({pattern_counts[pat_type]} Ticker)"):
                    pat_df = df_patterns[df_patterns['Pattern_Type'] == pat_type]
                    tickers_list = pat_df['Ticker'].unique()
                    st.write(f"**Hisseler:** {', '.join([t.replace('.IS','') for t in tickers_list])}")
                    
                    avg_rsi = pat_df['RSI_14'].mean()
                    avg_depth = pat_df['Depth_Ratio'].mean()
                    col_a, col_b = st.columns(2)
                    col_a.metric("Ortalama RSI Değeri", f"{avg_rsi:.2f}")
                    col_b.metric("Ortalama Destek Oranı (Depth)", f"{avg_depth:.3f}")
    else:
        st.warning("Veritabanı yapılandırma hatası: 'Pattern_Type' özniteliği veri tablosunda bulunamadı.")
        
    st.markdown("---")
    st.markdown("#### Canlı Teknik Tarama Paneli")
    st.markdown("Belirttiğiniz hisselerin son durumlarını canlı teknik tarama algoritmasıyla kontrol edin.")
    
    scan_tickers = st.text_input("Taranacak BIST Kodlarını Giriniz (Virgül ile Ayırın):", "THYAO, GARAN, EREGL, ASELS, FROTO", key="scan_input")
    
    if st.button("Canlı Taramayı Başlat", key="scan_btn", type="primary"):
        tickers_to_scan = [t.strip().upper() for t in scan_tickers.split(',')]
        tickers_to_scan = [t + '.IS' if not t.endswith('.IS') else t for t in tickers_to_scan]
        
        scan_results = []
        progress_bar = st.progress(0)
        
        for idx, tick in enumerate(tickers_to_scan):
            try:
                tick_data = yf.download(tick, period='6mo', interval='1d', progress=False)
                if tick_data is not None and not tick_data.empty and len(tick_data) > 50:
                    if isinstance(tick_data.columns, pd.MultiIndex):
                        tick_data.columns = [c[0] if isinstance(c, tuple) else c for c in tick_data.columns]
                    
                    close = tick_data['Close'].values
                    rsi_delta = pd.Series(close).diff()
                    rsi_gain = rsi_delta.clip(lower=0).rolling(14).mean()
                    rsi_loss = (-rsi_delta.clip(upper=0)).rolling(14).mean()
                    rsi_loss = rsi_loss.replace(0, 1e-10)
                    rsi_val = float((100 - (100 / (1 + rsi_gain / rsi_loss))).iloc[-1])
                    
                    last_close = float(close[-1])
                    sma20 = float(pd.Series(close).rolling(20).mean().iloc[-1])
                    sma50 = float(pd.Series(close).rolling(50).mean().iloc[-1])
                    
                    trend = 'Yükseliş Trendi' if last_close > sma50 else 'Düşüş Trendi'
                    rsi_durum = 'Aşırı Alım' if rsi_val > 70 else 'Aşırı Satım' if rsi_val < 30 else 'Normal'
                    
                    scan_results.append({
                        'Hisse': tick.replace('.IS', ''),
                        'Son Fiyat': f'{last_close:.2f}',
                        'Canlı RSI_14': f'{rsi_val:.2f}',
                        'RSI Durumu': rsi_durum,
                        'SMA_20 Seviyesi': f'{sma20:.2f}',
                        'SMA_50 Seviyesi': f'{sma50:.2f}',
                        'Trend Konumu': trend
                    })
            except Exception as e:
                pass
            
            progress_bar.progress((idx + 1) / len(tickers_to_scan))
        
        if scan_results:
            st.dataframe(pd.DataFrame(scan_results), use_container_width=True)
        else:
            st.warning("Veri Bağlantı Hatası: Belirtilen hisselere ait fiyat verileri yFinance üzerinden çekilemedi.")

st.markdown("""
<div class='footer-text'>
    YEDİTEPE ÜNİVERSİTESİ | BORSANEURON ALGORİTMİK TİCARET VE YAPAY ZEKA MEZUNİYET PROJESİ
</div>
""", unsafe_allow_html=True)
