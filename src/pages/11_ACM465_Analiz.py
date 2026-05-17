import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate
from sklearn.metrics import accuracy_score, classification_report, f1_score
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

st.markdown("<div class='brand-header'>ACM 465 | VERİ MADENCİLİĞİ & YAPAY ZEKA PROJESİ</div>", unsafe_allow_html=True)
st.markdown("**Soru:** Bir hisse senedinin 5 gün sonraki kapanış fiyatı bugünkünden yüksek mi olacak?")

# --- Veri Yükleme ---
@st.cache_data(ttl=3600)
def load_data():
    paths = [
        "bist_ai_dataset_real_30cols.csv",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'bist_ai_dataset_real_30cols.csv'),
        "/mount/src/borsaneuron/bist_ai_dataset_real_30cols.csv"
    ]
    for p in paths:
        if os.path.exists(p):
            return pd.read_csv(p)
    return None

df = load_data()

if df is None:
    st.error("Veriseti bulunamadı!")
    st.stop()

# Sekmeler
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "1. Veri Keşfi",
    "2. Korelasyon & Eleme",
    "3. K-Means & PCA",
    "4. Model Karşılaştırması",
    "5. Zaman Serisi (Prophet)",
    "6. Finansal Backtest"
])

# ==========================================
# TAB 1: VERİ KEŞFİ (DESCRIPTIVE STATISTICS)
# ==========================================
with tab1:
    st.markdown("### Veri Keşfi ve Tanımlayıcı İstatistikler")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Toplam Satır", f"{df.shape[0]:,}")
    col2.metric("Toplam Sütun", f"{df.shape[1]}")
    col3.metric("Missing Value", "0" if not df.isnull().values.any() else f"{df.isnull().sum().sum()}")
    
    st.markdown("#### Veri Setinin İlk 5 Satırı")
    st.dataframe(df.head(), use_container_width=True)
    
    st.markdown("#### Descriptive Statistics (Tanımlayıcı İstatistikler)")
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    desc = df[num_cols].describe().T
    st.dataframe(desc, use_container_width=True)
    
    # Target dağılımı
    st.markdown("#### Hedef Değişken (Target_T5) Dağılımı")
    target_counts = df['Target_T5'].value_counts()
    fig_target = px.pie(values=target_counts.values, names=['Düşüş (0)', 'Yükseliş (1)'],
                        title="Target_T5 Sınıf Dağılımı",
                        color_discrete_sequence=['#ff4444', '#00ff88'])
    fig_target.update_layout(template="plotly_dark", plot_bgcolor="#0e1117", paper_bgcolor="#0e1117")
    st.plotly_chart(fig_target, use_container_width=True)
    
    # RSI histogramı
    st.markdown("#### RSI_14 Dağılımı")
    fig_rsi = px.histogram(df, x='RSI_14', nbins=50, title="RSI_14 Histogram",
                           color_discrete_sequence=['#00f2ff'])
    fig_rsi.update_layout(template="plotly_dark", plot_bgcolor="#0e1117", paper_bgcolor="#0e1117")
    st.plotly_chart(fig_rsi, use_container_width=True)
    
    # Target'a göre ortalamalar
    st.markdown("#### Target'a Göre İndikatör Ortalamaları")
    indicator_cols = ['RSI_14', 'MACD', 'ATR_14', 'Stoch_K', 'Volume_Trend']
    target_means = df.groupby('Target_T5')[indicator_cols].mean()
    st.dataframe(target_means, use_container_width=True)

# ==========================================
# TAB 2: KORELASYON ANALİZİ
# ==========================================
with tab2:
    st.markdown("### Korelasyon Analizi ve Yüksek Korelasyonlu Değişkenlerin Elenmesi")
    
    with st.expander("📌 Neden Yüksek Korelasyonlu Değişkenler Eleniyor?", expanded=True):
        st.markdown('''
        * Yüksek korelasyonlu (>0.90) değişkenler aynı bilgiyi tekrar taşır ve **overfitting** riskini artırır.
        * Korelasyon matrisinin üst üçgeni alınıp, 0.90 üzeri korelasyona sahip sütunlar elenir.
        * Bu sayede model daha genellenebilir ve daha az gürültüye maruz kalır.
        ''')
    
    sensor_cols = df.select_dtypes(include=[np.number]).drop(
        columns=['Target_T3', 'Target_T5', 'Target_T15', 'Max_Drawdown_15D', 'Max_Gain_15D'], errors='ignore')
    corr = sensor_cols.corr()
    
    fig_corr = px.imshow(corr, text_auto=False, aspect="auto",
                         title="Korelasyon Isı Haritası (Heatmap)",
                         color_continuous_scale="RdBu_r")
    fig_corr.update_layout(template="plotly_dark", plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
                           width=900, height=700)
    st.plotly_chart(fig_corr, use_container_width=True)
    
    # Eleme
    cor_matrix = sensor_cols.corr().abs()
    upper = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(bool))
    drop_list = [col for col in upper.columns if any(upper[col] > 0.90)]
    
    st.markdown(f"#### 0.90 Üzeri Korelasyonlu Elenen Değişkenler ({len(drop_list)} adet)")
    st.code(str(drop_list))
    
    remaining = [c for c in sensor_cols.columns if c not in drop_list]
    st.markdown(f"#### Kalan Değişkenler ({len(remaining)} adet)")
    st.code(str(remaining))

# ==========================================
# TAB 3: K-MEANS & PCA
# ==========================================
with tab3:
    st.markdown("### K-Means Kümeleme ve PCA Görselleştirmesi")
    
    with st.expander("📌 Metodoloji", expanded=True):
        st.markdown('''
        * **K-Means:** `init='k-means++'`, Öklid mesafesi ile kümeleme. Optimum k değeri Elbow metodu ile belirlenir.
        * **PCA:** Yüksek boyutlu veri 2 bileşene indirgenerek küme ayrışması görselleştirilir.
        * **Teknik Analiz İlişkisi:** Yüksek ATR kümesinde Bayrak/Kırılım, düşük RSI kümesinde TOBO/Çift Dip aranır.
        ''')
    
    k_clusters = st.slider("Küme Sayısı (K)", min_value=2, max_value=8, value=5)
    
    # Hisse bazlı istatistikler
    ticker_stats = df.groupby('Ticker')[['ATR_14', 'Max_Drawdown_15D', 'Max_Gain_15D', 'RSI_14']].mean().dropna()
    scaler_km = StandardScaler()
    scaled_tickers = scaler_km.fit_transform(ticker_stats)
    
    kmeans = KMeans(n_clusters=k_clusters, random_state=42, n_init=10)
    ticker_stats['Cluster'] = kmeans.fit_predict(scaled_tickers)
    
    st.markdown(f"**Inertia:** {kmeans.inertia_:.2f}")
    
    # PCA 2D
    pca = PCA(n_components=2)
    pca_res = pca.fit_transform(scaled_tickers)
    ticker_stats['PCA1'] = pca_res[:, 0]
    ticker_stats['PCA2'] = pca_res[:, 1]
    
    st.markdown(f"**PCA Açıklanan Varyans:** PC1={pca.explained_variance_ratio_[0]*100:.1f}%, PC2={pca.explained_variance_ratio_[1]*100:.1f}%")
    
    fig2 = px.scatter(
        ticker_stats.reset_index(), x="PCA1", y="PCA2",
        color="Cluster", hover_data=["Ticker", "ATR_14", "Max_Gain_15D"],
        title="K-Means Kümeleri - PCA 2D Görselleştirme",
        color_continuous_scale="Turbo"
    )
    fig2.update_traces(marker=dict(size=12, opacity=0.8, line=dict(width=1, color='DarkSlateGrey')))
    fig2.update_layout(template="plotly_dark", plot_bgcolor="#0e1117", paper_bgcolor="#0e1117")
    st.plotly_chart(fig2, use_container_width=True)
    
    st.markdown("#### Küme Merkezleri (Ortalama Değerler)")
    st.dataframe(ticker_stats.groupby('Cluster')[['ATR_14', 'Max_Drawdown_15D', 'Max_Gain_15D', 'RSI_14']].mean())
    
    # Kümülatif varyans grafiği
    pca_full = PCA()
    pca_full.fit(scaled_tickers)
    cumvar = np.cumsum(pca_full.explained_variance_ratio_)
    fig_var = px.line(x=range(1, len(cumvar)+1), y=cumvar,
                      title="PCA Kümülatif Açıklanan Varyans",
                      labels={'x': 'Bileşen Sayısı', 'y': 'Kümülatif Varyans'})
    fig_var.update_layout(template="plotly_dark", plot_bgcolor="#0e1117", paper_bgcolor="#0e1117")
    st.plotly_chart(fig_var, use_container_width=True)

# ==========================================
# TAB 4: MODEL KARŞILAŞTIRMASI (K-NN + RF + ANN)
# ==========================================
with tab4:
    st.markdown("### Yapay Zeka Algoritmaları Karşılaştırması")
    st.markdown("K-NN, Random Forest ve Yapay Sinir Ağı (ANN) modellerinin karşılaştırmalı analizi.")
    
    with st.expander("📌 Model Parametre Detayları", expanded=True):
        st.markdown('''
        * **K-NN:** GridSearchCV ile optimize edilen k değeri, Öklid mesafesi.
        * **Random Forest:** `n_estimators=100`, `criterion='gini'`, `random_state=42`.
        * **ANN (MLP):** `hidden_layer_sizes=(64, 32, 16)`, `activation='relu'`, `solver='adam'`, `max_iter=100`.
        * **Ölçeklendirme:** StandardScaler ile Z-score normalizasyonu.
        ''')
    
    features = ['RSI_14', 'MACD', 'ATR_14', 'Stoch_K', 'Volume_Trend', 'Depth_Ratio', 'Neckline_Slope', 'Expert_Signal']
    df_ml = df.dropna(subset=features + ['Target_T5']).copy()
    
    if st.button("Modelleri Eğit ve Karşılaştır", key="train_btn"):
        with st.spinner("3 model eğitiliyor ve GridSearchCV çalışıyor..."):
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
            
            # ANN
            ann = MLPClassifier(hidden_layer_sizes=(64, 32, 16), activation='relu',
                                solver='adam', max_iter=100, batch_size=128, random_state=42, verbose=False)
            ann.fit(X_train_s, y_train)
            ann_pred = ann.predict(X_test_s)
            ann_acc = accuracy_score(y_test, ann_pred)
            ann_f1 = f1_score(y_test, ann_pred)
            
            # Metrik kartları
            col1, col2, col3 = st.columns(3)
            for col, name, acc, f1v in [(col1, f"K-NN (k={best_k})", knn_acc, knn_f1),
                                         (col2, "Random Forest", rf_acc, rf_f1),
                                         (col3, "ANN (MLP)", ann_acc, ann_f1)]:
                with col:
                    st.markdown("<div class='terminal-card'>", unsafe_allow_html=True)
                    st.markdown(f"<div class='metric-label'>{name}</div>", unsafe_allow_html=True)
                    st.markdown(f"<div class='metric-value'>%{acc*100:.1f}</div>", unsafe_allow_html=True)
                    st.markdown(f"<small>F1-Score: {f1v:.4f}</small>", unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)
            
            # Karşılaştırma tablosu
            results_df = pd.DataFrame({
                'Model': [f'K-NN (k={best_k})', 'Random Forest', 'ANN (MLP)'],
                'Accuracy': [knn_acc, rf_acc, ann_acc],
                'F1-Score': [knn_f1, rf_f1, ann_f1]
            })
            
            fig_comp = px.bar(results_df, x='Model', y=['Accuracy', 'F1-Score'],
                              barmode='group', title="Model Performans Karşılaştırması",
                              color_discrete_sequence=['#00f2ff', '#00ff88'])
            fig_comp.update_layout(template="plotly_dark", plot_bgcolor="#0e1117", paper_bgcolor="#0e1117")
            st.plotly_chart(fig_comp, use_container_width=True)
            
            # Feature Importance
            importances = rf.feature_importances_
            fig_fi = px.bar(x=importances, y=features, orientation='h',
                            title="Random Forest - Feature Importance (Özellik Önem Derecesi)",
                            color=importances, color_continuous_scale="Viridis")
            fig_fi.update_layout(template="plotly_dark", plot_bgcolor="#0e1117", paper_bgcolor="#0e1117")
            st.plotly_chart(fig_fi, use_container_width=True)
            
            st.info(f"💡 GridSearchCV sonucu en iyi K-NN k değeri: **{best_k}** (CV Accuracy: {knn_gs.best_score_:.4f})")

# ==========================================
# TAB 5: ZAMAN SERİSİ (PROPHET)
# ==========================================
with tab5:
    st.markdown("### Prophet ile Gelecek Fiyat Tahmini")
    
    with st.expander("📌 Algoritma Detayları", expanded=True):
        st.markdown('''
        * **Meta Prophet:** Toplanabilir (additive) regresyon modeli. `yearly_seasonality=True`, `daily_seasonality=False`.
        * Teknik Analizdeki yeri: Klasik TA anlık kırılımlar verirken, Prophet makro trendi gösterir.
        ''')
    
    if Prophet is None:
        st.error("Prophet kütüphanesi kurulu değil!")
    else:
        selected_ticker = st.selectbox("Hisse Seçin", df['Ticker'].unique()[:50], index=0)
        days_ahead = st.slider("Kaç gün ileri tahmin?", 10, 90, 60)
        
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
                
                fig3 = go.Figure()
                fig3.add_trace(go.Scatter(x=df_prophet['ds'], y=df_prophet['y'], mode='lines', name='Gerçek Fiyat', line=dict(color='#00ff88')))
                fig3.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Prophet Tahmini', line=dict(color='#ffbf00')))
                fig3.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], fill=None, mode='lines', line_color='rgba(255,191,0,0)', showlegend=False))
                fig3.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], fill='tonexty', mode='lines', fillcolor='rgba(255,191,0,0.2)', line_color='rgba(255,191,0,0)', name='Güven Aralığı'))
                fig3.update_layout(title=f"{selected_ticker} - {days_ahead} Günlük Trend Tahmini",
                                   template="plotly_dark", plot_bgcolor="#0e1117", paper_bgcolor="#0e1117", hovermode="x unified")
                st.plotly_chart(fig3, use_container_width=True)

# ==========================================
# TAB 6: FİNANSAL BACKTEST
# ==========================================
with tab6:
    st.markdown("### Finansal Backtest: Model Para Kazandırıyor mu?")
    
    with st.expander("📌 Backtest Parametreleri", expanded=True):
        st.markdown('''
        * **Kronolojik Bölme:** %80 Train, %20 Test (Data Leakage koruması).
        * **Strateji:** RF modeli "Yükselecek" dediğinde al, aksi halde bekle.
        * **Muhafazakâr Varsayım:** Kazancın %30'u yakalanır, kayıpların %50'si yansır.
        ''')
    
    if st.button("Backtest Başlat", key="backtest_btn"):
        with st.spinner("Simülasyon çalışıyor..."):
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
            fig4.add_trace(go.Scatter(x=daily_net.index, y=daily_net['Portfoy_Degeri'], mode='lines', name='AI Stratejisi', line=dict(color='#00ff88', width=3)))
            fig4.add_trace(go.Scatter(x=daily_net.index, y=daily_net['Buy_And_Hold'], mode='lines', name='Buy & Hold', line=dict(color='#ff4444', dash='dash')))
            fig4.update_layout(title="AI Portföy Büyümesi (Out-of-Sample Backtest)", yaxis_title="Sermaye (TL)",
                               template="plotly_dark", plot_bgcolor="#0e1117", paper_bgcolor="#0e1117")
            st.plotly_chart(fig4, use_container_width=True)
            
            final_capital = daily_net['Portfoy_Degeri'].iloc[-1]
            net_profit = ((final_capital - 100000) / 100000) * 100
            win_rate = (test_df[test_df['AI_Signal'] == 1]['Target_T5'].mean()) * 100
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Başlangıç", "100,000 ₺")
            col2.metric("Bitiş", f"{final_capital:,.2f} ₺", f"%{net_profit:.1f}")
            col3.metric("Win Rate", f"%{win_rate:.1f}")
