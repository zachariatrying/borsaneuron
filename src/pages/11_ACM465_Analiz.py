import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os

try:
    from prophet import Prophet
except ImportError:
    Prophet = None

# --- Sayfa Yapılandırması ---
st.set_page_config(
    page_title="BORSANEURON | VERİ MADENCİLİĞİ",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS ---
TERMINAL_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto+Mono:wght@400;700&family=Inter:wght@400;700&display=swap');
    
    .stApp { background-color: #0e1117; color: #e2e8f0; font-family: 'Inter', sans-serif; }
    
    .terminal-card {
        background-color: #1a1c23;
        border: 1px solid #2d3748;
        padding: 24px;
        margin-bottom: 20px;
        border-radius: 4px;
    }
    
    .brand-header {
        color: #00f2ff;
        font-family: 'Roboto Mono', monospace;
        font-weight: 700;
        letter-spacing: 2px;
        font-size: 1.6rem;
        border-bottom: 2px solid #00f2ff;
        padding-bottom: 10px;
        margin-bottom: 20px;
    }
    
    .metric-value { color: #00ff88; font-weight: bold; font-size: 1.5rem; font-family: 'Roboto Mono', monospace; }
    .metric-label { font-size: 0.8rem; color: #94a3b8; text-transform: uppercase; margin-bottom: 4px;}
</style>
"""
st.markdown(TERMINAL_CSS, unsafe_allow_html=True)

st.markdown("<div class='brand-header'>ACM 465 | VERİ MADENCİLİĞİ İŞGÖRÜLERİ</div>", unsafe_allow_html=True)

# --- Veri Yükleme ---
@st.cache_data
def load_data():
    # Use relative path so it works on Streamlit Cloud
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Go up two levels from src/pages to root
    dataset_path = os.path.join(current_dir, '..', '..', 'bist_ai_dataset_real_30cols.csv')
    
    if os.path.exists(dataset_path):
        return pd.read_csv(dataset_path)
    return None

df = load_data()

if df is None:
    st.error("Veriseti bulunamadı!")
    st.stop()

# Sekmeler
tab1, tab2, tab3 = st.tabs(["1. Model Karşılaştırması", "2. Hisse Segmentasyonu (K-Means)", "3. Zaman Serisi (Prophet)"])

# ==========================================
# TAB 1: MODEL KARŞILAŞTIRMASI & FEATURE IMPORTANCE
# ==========================================
with tab1:
    st.markdown("### Algoritma Performans Analizi (Regresyon / Sınıflandırma)")
    st.markdown("Model: BIST Hisseleri için **Target_T5 (5 Günlük Getiri Hedefi)** tahmini.")
    
    features = ['RSI_14', 'MACD', 'ATR_14', 'Stoch_K', 'Volume_Trend', 'Depth_Ratio', 'Neckline_Slope', 'Expert_Signal']
    df_ml = df.dropna(subset=features + ['Target_T5']).copy()
    
    if st.button("Modelleri Eğit ve Karşılaştır", key="train_btn"):
        with st.spinner("Modeller eğitiliyor..."):
            X = df_ml[features]
            y = df_ml['Target_T5']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Random Forest
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            rf.fit(X_train_scaled, y_train)
            rf_acc = accuracy_score(y_test, rf.predict(X_test_scaled))
            
            # ANN
            ann = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=200, random_state=42)
            ann.fit(X_train_scaled, y_train)
            ann_acc = accuracy_score(y_test, ann.predict(X_test_scaled))
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("<div class='terminal-card'>", unsafe_allow_html=True)
                st.markdown("<div class='metric-label'>Random Forest Doğruluğu</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='metric-value'>%{rf_acc*100:.1f}</div>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
            with col2:
                st.markdown("<div class='terminal-card'>", unsafe_allow_html=True)
                st.markdown("<div class='metric-label'>Yapay Sinir Ağı (ANN) Doğruluğu</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='metric-value'>%{ann_acc*100:.1f}</div>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
            
            st.info("💡 **İşgörü:** Borsa gibi gürültülü verilerde Karar Ağaçları (Random Forest), Yapay Sinir Ağlarına göre overfitting'e daha az yatkındır ve daha istikrarlı sonuçlar verebilir.")
            
            # Feature Importance
            importances = rf.feature_importances_
            fig = px.bar(
                x=importances, y=features, 
                orientation='h', 
                title="Random Forest - Özellik Önem Derecesi (Feature Importance)",
                color=importances, color_continuous_scale="Viridis"
            )
            fig.update_layout(template="plotly_dark", plot_bgcolor="#0e1117", paper_bgcolor="#0e1117")
            st.plotly_chart(fig, use_container_width=True)

# ==========================================
# TAB 2: HİSSE SEGMENTASYONU (K-MEANS)
# ==========================================
with tab2:
    st.markdown("### K-Means ile Hisse Profili Kümeleme")
    st.markdown("Hisseleri **Volatilite (ATR), Risk (Max Drawdown) ve Potansiyel (Max Gain)** metriklerine göre kümeliyoruz.")
    
    k_clusters = st.slider("Küme Sayısı (K)", min_value=2, max_value=6, value=3)
    
    ticker_stats = df.groupby('Ticker')[['ATR_14', 'Max_Drawdown_15D', 'Max_Gain_15D', 'RSI_14']].mean().dropna()
    scaler_km = StandardScaler()
    scaled_tickers = scaler_km.fit_transform(ticker_stats)
    
    kmeans = KMeans(n_clusters=k_clusters, random_state=42)
    ticker_stats['Cluster'] = kmeans.fit_predict(scaled_tickers)
    
    # PCA
    pca = PCA(n_components=2)
    pca_res = pca.fit_transform(scaled_tickers)
    ticker_stats['PCA1'] = pca_res[:, 0]
    ticker_stats['PCA2'] = pca_res[:, 1]
    
    fig2 = px.scatter(
        ticker_stats.reset_index(), 
        x="PCA1", y="PCA2", 
        color="Cluster", hover_data=["Ticker", "ATR_14", "Max_Gain_15D"],
        title="PCA ile Kümelerin Görselleştirilmesi (Cluster Distribution)",
        color_continuous_scale="Turbo"
    )
    fig2.update_traces(marker=dict(size=10, opacity=0.8, line=dict(width=1, color='DarkSlateGrey')))
    fig2.update_layout(template="plotly_dark", plot_bgcolor="#0e1117", paper_bgcolor="#0e1117")
    st.plotly_chart(fig2, use_container_width=True)
    
    st.markdown("#### Küme Merkezleri (Ortalama Değerler)")
    st.dataframe(ticker_stats.groupby('Cluster')[['ATR_14', 'Max_Drawdown_15D', 'Max_Gain_15D', 'RSI_14']].mean().style.background_gradient(cmap='viridis'))
    
    st.info("💡 **Stratejik Kural (If-Else):** Eğer hisse Yüksek Kazanç/Yüksek Risk kümesindeyse (Cluster 0) -> Sıkı STOP LOSS kullan. Eğer Düşük Risk/Dengeli kümedeyse (Cluster 2) -> Uzun vadeli ANN tahminine güven.")

# ==========================================
# TAB 3: ZAMAN SERİSİ (PROPHET)
# ==========================================
with tab3:
    st.markdown("### Prophet ile Gelecek Fiyat Tahmini")
    st.markdown("Meta (Facebook) Prophet algoritması ile zaman serisi trendini ve mevsimselliği modelleyerek gelecek projeksiyonu oluşturuyoruz.")
    
    if Prophet is None:
        st.error("Prophet kütüphanesi kurulu değil!")
    else:
        selected_ticker = st.selectbox("Hisse Seçin", df['Ticker'].unique()[:50], index=0)
        days_ahead = st.slider("Kaç gün ileri tahmin edilsin?", 10, 90, 60)
        
        if st.button("Tahmini Başlat", key="prophet_btn"):
            with st.spinner("Prophet modeli eğitiliyor..."):
                df_ticker = df[df['Ticker'] == selected_ticker].copy()
                df_ticker['Date'] = pd.to_datetime(df_ticker['Date'])
                df_ticker = df_ticker.sort_values('Date')
                
                df_prophet = df_ticker[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})
                
                m = Prophet(daily_seasonality=False, yearly_seasonality=True)
                m.fit(df_prophet)
                
                future = m.make_future_dataframe(periods=days_ahead)
                forecast = m.predict(future)
                
                # Plotly ile Prophet Grafiği
                fig3 = go.Figure()
                # Geçmiş Veri
                fig3.add_trace(go.Scatter(x=df_prophet['ds'], y=df_prophet['y'], mode='lines', name='Gerçek Fiyat', line=dict(color='#00ff88')))
                # Tahmin
                fig3.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Prophet Tahmini', line=dict(color='#ffbf00')))
                # Güven Aralığı
                fig3.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], fill=None, mode='lines', line_color='rgba(255,191,0,0)', showlegend=False))
                fig3.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], fill='tonexty', mode='lines', fillcolor='rgba(255,191,0,0.2)', line_color='rgba(255,191,0,0)', name='Güven Aralığı'))
                
                fig3.update_layout(
                    title=f"{selected_ticker} - Gelecek {days_ahead} Günlük Trend Tahmini",
                    template="plotly_dark", plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
                    hovermode="x unified"
                )
                st.plotly_chart(fig3, use_container_width=True)
                
                st.success(f"{selected_ticker} için önümüzdeki {days_ahead} günlük satış/fiyat projeksiyonu başarıyla oluşturuldu.")
