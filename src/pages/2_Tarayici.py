"""
BORSANEURON | PROFESSIONAL FINANCE TERMINAL
Bloomberg-Style Analytics & Pattern Recognition
"""
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import json
import sys, os
import time
import pytz
from datetime import datetime, timedelta

# Proje kok dizini
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from analyzer import Analyzer

# Zaman Ayari (Europe/Istanbul)
TR_TZ = pytz.timezone('Europe/Istanbul')

def get_system_time():
    return datetime.now(TR_TZ).strftime('%H:%M:%S')

# Sayfa Yapilandirmasi
st.set_page_config(
    page_title="BORSANEURON | TERMINAL",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional Terminal CSS (No Emojis)
TERMINAL_STYLE = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&family=Roboto+Mono:wght@400;700&display=swap');
    
    .stApp { background-color: #0e1117; color: #e2e8f0; font-family: 'Inter', sans-serif; }
    
    /* Terminal Card */
    .terminal-card {
        background-color: #1a1c23;
        border: 1px solid #2d3748;
        padding: 24px;
        margin-bottom: 20px;
        border-radius: 4px;
    }
    
    /* Header & Metrics */
    .brand-header {
        color: #00f2ff;
        font-family: 'Roboto Mono', monospace;
        font-weight: 700;
        letter-spacing: 2px;
        font-size: 1.5rem;
        border-bottom: 1px solid #2d3748;
        padding-bottom: 10px;
        margin-bottom: 20px;
    }
    
    .metric-box {
        display: inline-block;
        margin-right: 15px;
        padding: 5px 12px;
        border-radius: 2px;
        background: #0e1117;
        border: 1px solid #2d3748;
    }
    
    .metric-label { font-size: 0.7rem; color: #94a3b8; text-transform: uppercase; }
    .metric-value-cyan { color: #00f2ff; font-weight: bold; font-family: 'Roboto Mono', monospace; }
    .metric-value-amber { color: #ffbf00; font-weight: bold; font-family: 'Roboto Mono', monospace; }
    
    /* Buttons & Inputs */
    .stButton > button {
        background-color: #1a1c23; color: #00f2ff; border: 1px solid #00f2ff;
        border-radius: 2px; text-transform: uppercase; font-weight: 600;
    }
    .stButton > button:hover { background-color: #00f2ff; color: #0e1117; }
    
    /* Sidebar */
    [data-testid="stSidebar"] { background-color: #0e1117; border-right: 1px solid #2d3748; }
</style>
"""
st.markdown(TERMINAL_STYLE, unsafe_allow_html=True)

# Endpoint Configuration
AI_ENDPOINT = "https://borsaneuron-api.onrender.com/predict"

# --- Hardcoded Index Lists ---
BIST30 = "AKBNK,ARCLK,ASELS,ASTOR,BIMAS,BRSAN,EKGYO,ENKAI,EREGL,FROTO,GARAN,GUBRF,HEKTS,ISCTR,KCHOL,KONTR,KOZAL,KRDMD,ODAS,OYAKC,PETKM,PGSUS,SAHOL,SASA,SISE,TCELL,THYAO,TOASO,TUPRS,YKBNK".split(',')
BIST100 = "AEFES,AGHOL,AKBNK,AKCNS,AKGRT,AKSEN,ALARK,ALBRK,ALEFS,ALGYO,ALKIM,ARCLK,ARDYZ,ASELS,ASTOR,ASUZU,AYDEM,BAGFS,BERA,BIENY,BIMAS,BRSAN,BRYAT,BUCIM,CANTE,CCOLA,CIMSA,CWENE,DOAS,DOHOL,EGEEN,EKGYO,ENJSA,ENKAI,EREGL,EUPWR,FROTO,GARAN,GENIL,GESAN,GLYHO,GUBRF,GWIND,HALKB,HEKTS,IPEKE,ISCTR,ISDMR,ISGYO,ISMEN,IZMDC,KARDM,KAYSE,KCHOL,KCAER,KFEIN,KONTR,KORDS,KOZAA,KOZAL,KRDMD,MAVI,MGROS,MIATK,NETAS,ODAS,OTKAR,OYAKC,PENTA,PETKM,PGSUS,QUAGR,SAHOL,SASA,SAYAS,SDTTR,SISE,SKBNK,SMRTG,SOKM,TABGD,TAVHL,TCELL,THYAO,TKFEN,TKNSA,TMSN,TOASO,TSKB,TTKOM,TTRAK,TUPRS,TURSG,ULKER,VAKBN,VESBE,VESTL,YEOTK,YKBNK,ZOREN".split(',')

# --- Session State ---
if 'watchlist' not in st.session_state: st.session_state.watchlist = []
if 'last_results' not in st.session_state: st.session_state.last_results = []

# --- Helper Functions ---
@st.cache_resource
def get_analyzer():
    return Analyzer()

analyzer_instance = get_analyzer()

def calculate_perf(df, lookback_bars):
    if len(df) < lookback_bars + 1: return 0.0
    current = df.iloc[-1]['Close']
    past = df.iloc[-(lookback_bars+1)]['Close']
    return float((current - past) / past)

@st.cache_data(ttl=600)
def fetch_data(ticker, interval, period):
    try:
        session = requests.Session()
        session.headers.update({'User-Agent': 'Mozilla/5.0'})
        symbol = f"{ticker}.IS"
        df = yf.download(symbol, period=period, interval=interval, progress=False, session=session)
        
        if df is None or df.empty or len(df) < 50: return None
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        
        df = df.rename(columns={'Open':'Open', 'High':'High', 'Low':'Low', 'Close':'Close', 'Volume':'Volume'})
        if df['Close'].isnull().any(): df = df.ffill().dropna()
        
        return df
    except:
        return None

def get_ai_prediction(hisse, df):
    try:
        # Prepare 100 bars for AI
        df_sub = df.tail(100).copy()
        df_sub['date'] = df_sub.index.strftime('%Y-%m-%d')
        veriler = df_sub.rename(columns=lambda x: x.lower()).to_dict(orient='records')
        
        payload = {"hisse": hisse, "veriler": veriler}
        response = requests.post(AI_ENDPOINT, json=payload, timeout=12)
        
        if response.status_code == 200: return response.json()
        return {"hata": "Yapay Zeka Onayı Bekleniyor"}
    except:
        return {"hata": "Yapay Zeka Onayı Bekleniyor"}

def execute_analysis(df, active_patterns, label):
    results = []
    df_c = df.copy()
    if 'Date' not in df_c.columns: df_c['Date'] = df_c.index
    
    # Configure Patterns
    analyzer_instance.config['enabled_patterns'] = {
        'tobo': "TOBO" in active_patterns,
        'obo': "OBO" in active_patterns,
        'cup': "Fincan Kulp" in active_patterns,
        'flag': "Boğa Bayrağı" in active_patterns,
        'flama': "Flama" in active_patterns,
    }
    
    try:
        tf = "Gunluk" if "GUNLUK" in label else "Saatlik"
        df_full = analyzer_instance.add_indicators(df_c)
        found = analyzer_instance.detect_classic_patterns(df_full, timeframe=tf)
        
        if "RSI Uyumsuzluğu" in active_patterns:
            zz = analyzer_instance.calculate_zigzag(df_full)
            found.extend(analyzer_instance.detect_rsi_divergence(df_full, zz, tf))
        
        if "High Tight Flag (Roket)" in active_patterns:
            found.extend(analyzer_instance.detect_high_tight_flag(df_full))
            
        if "Mum Formasyonları" in active_patterns:
            found.extend(analyzer_instance.detect_candlestick_patterns(df_full, tf))
            
        for p in found:
            results.append({
                "Name": p.get('name', 'Bilinmeyen'),
                "Score": p.get('score', 0),
                "Signal": p.get('signal', 'Bullish'),
                "Target": p.get('target', 0),
                "Stop": p.get('stop', 0),
                "Desc": p.get('desc', '')
            })
    except: pass
    return results

# --- Header ---
c_title, c_clock = st.columns([3, 1])
with c_title:
    st.markdown("<div class='brand-header'>BORSANEURON | TERMINAL</div>", unsafe_allow_html=True)
with c_clock:
    st.markdown(f"<div style='text-align:right; color:#94a3b8; font-family:Roboto Mono;'>SISTEM SAATI: {get_system_time()}</div>", unsafe_allow_html=True)

# --- Sidebar ---
with st.sidebar:
    st.markdown("<div style='color:#00f2ff; font-weight:bold; padding-bottom:10px;'>KONTROL MERKEZI</div>", unsafe_allow_html=True)
    
    tarama_modu = st.radio("Tarama Listesi", ["BIST 30", "BIST 100", "OZEL TAKIP LISTESI"])
    
    if tarama_modu == "OZEL TAKIP LISTESI":
        st.info("Virgül kullanarak hisse ekleyin (örn: THYAO, ASELS)")
        new_tickers = st.text_input("Hisse Ekle/Guncelle", ",".join(st.session_state.watchlist))
        if st.button("LISTEYI KAYDET"):
            st.session_state.watchlist = [t.strip().upper() for t in new_tickers.split(',') if t.strip()]
            st.success("Liste Guncellendi")
        current_hisseler = st.session_state.watchlist
    elif tarama_modu == "BIST 30":
        current_hisseler = BIST30
    else:
        current_hisseler = BIST100

    zaman_secimi = st.selectbox("Periyot", ["GUNLUK (1D)", "SAATLIK (1h)"])
    if "GUNLUK" in zaman_secimi: yf_int, yf_per, label = "1d", "2y", "GUNLUK"
    else: yf_int, yf_per, label = "60m", "730d", "SAATLIK"
    
    formasyon_secimi = st.multiselect("Formasyonlar", [
        "TOBO", "OBO", "Fincan Kulp", "Boğa Bayrağı", "Flama", 
        "High Tight Flag (Roket)", "RSI Uyumsuzluğu", "Mum Formasyonları"
    ], default=["Boğa Bayrağı", "TOBO", "Mum Formasyonları"])
    
    if st.button("TERMINAL TARAMASINI BASLAT", use_container_width=True):
        st.session_state.last_results = []
        progress = st.progress(0)
        for i, h in enumerate(current_hisseler):
            progress.progress((i+1)/len(current_hisseler))
            df = fetch_data(h, yf_int, yf_per)
            if df is not None:
                ai = get_ai_prediction(h, df)
                tech = execute_analysis(df, formasyon_secimi, label)
                
                # Performance Stats
                perf_h = calculate_perf(df, 1) # ~1 bar (hour if 1h, day if 1d)
                perf_m = calculate_perf(df, 22) # ~1 trading month
                
                st.session_state.last_results.append({
                    "hisse": h, "price": float(df.iloc[-1]['Close']), 
                    "perf_h": perf_h, "perf_m": perf_m,
                    "ai": ai, "tech": tech
                })
        progress.empty()

# --- Main Content ---
if not st.session_state.last_results:
    st.markdown("<div class='amber-text' style='text-align:center; padding:50px;'>Veri Bağlantısı Bekleniyor... Tarama başlatılmadı.</div>", unsafe_allow_html=True)
else:
    for res in st.session_state.last_results:
        with st.container():
            st.markdown("<div class='terminal-card'>", unsafe_allow_html=True)
            
            # Card Header
            c1, c2 = st.columns([3, 1])
            with c1:
                st.markdown(f"<span style='font-size:1.8rem; font-weight:700; color:#00f2ff;'>{res['hisse']}</span>", unsafe_allow_html=True)
                # Small Performance Metrics
                st.markdown(f"""
                <div class='metric-box'><span class='metric-label'>SAATLIK:</span> <span class='{'metric-value-cyan' if res['perf_h']>=0 else 'metric-value-amber'}'>%{res['perf_h']*100:.2f}</span></div>
                <div class='metric-box'><span class='metric-label'>AYLIK:</span> <span class='{'metric-value-cyan' if res['perf_m']>=0 else 'metric-value-amber'}'>%{res['perf_m']*100:.2f}</span></div>
                """, unsafe_allow_html=True)
            
            with c2:
                ai_data = res['ai']
                if 'hata' in ai_data:
                    st.markdown(f"<div class='metric-label'>AI DURUM</div><div class='metric-value-amber' style='font-size:0.9rem;'>{ai_data['hata']}</div>", unsafe_allow_html=True)
                else:
                    conf = ai_data.get('guven_orani', 0)
                    st.markdown(f"<div class='metric-label'>AI GUVENI</div><div class='metric-value-cyan' style='font-size:1.5rem;'>%{conf*100:.1f}</div>", unsafe_allow_html=True)
            
            # Patterns
            if res['tech']:
                st.markdown("<div style='margin-top:15px; border-top:1px solid #2d3748; padding-top:10px;'>", unsafe_allow_html=True)
                for p in res['tech']:
                    sig_color = "#00f2ff" if p['Signal'] == "Bullish" else "#ffbf00"
                    st.markdown(f"""
                    <div style='margin-bottom:5px;'>
                        <span style='background:{sig_color}; color:#0e1117; font-size:0.7rem; font-weight:bold; padding:2px 6px; border-radius:2px;'>{p['Name']}</span>
                        <span style='color:#94a3b8; font-size:0.8rem; margin-left:10px;'>{p['Desc']}</span>
                    </div>
                    """, unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
            
            # Prices
            p1, p2, p3 = st.columns(3)
            p1.metric("FIYAT", f"{res['price']:.2f}")
            p2.metric("HEDEF", f"{res['price']*1.05:.2f}", "5.0%")
            p3.metric("STOP", f"{res['price']*0.95:.2f}", "-5.0%", delta_color="inverse")
            
            st.markdown("</div>", unsafe_allow_html=True)
