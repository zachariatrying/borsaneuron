"""
BORSANEURON | PROFESSIONAL TERMINAL ONE-SHOT FIX
Comprehensive Market Analytics & Neural Inference Bridge
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
PROJECT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, '..'))
import analyzer
import importlib
importlib.reload(analyzer)
from analyzer import Analyzer
from live_inference_engine import LiveInferenceEngine

# Zaman Ayari (Europe/Istanbul)
TR_TZ = pytz.timezone('Europe/Istanbul')

def get_current_tr_time():
    return datetime.now(TR_TZ).strftime('%H:%M:%S')

# Sayfa Yapilandirmasi
st.set_page_config(
    page_title="BORSANEURON | TERMINAL",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Unified Terminal CSS (No Emojis)
TERMINAL_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto+Mono:wght@400;700&family=Inter:wght@400;700&display=swap');
    
    .stApp { background-color: #0e1117; color: #e2e8f0; font-family: 'Inter', sans-serif; }
    
    /* Terminal Card */
    .terminal-card {
        background-color: #1a1c23;
        border: 1px solid #2d3748;
        padding: 24px;
        margin-bottom: 20px;
        border-radius: 4px;
    }
    
    /* Branding */
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
    
    .metric-label { font-size: 0.7rem; color: #94a3b8; text-transform: uppercase; }
    .metric-value { color: #00f2ff; font-weight: bold; font-family: 'Roboto Mono', monospace; }
    
    /* Progress and Status */
    .status-msg {
        background: #1a1c23;
        color: #00f2ff;
        padding: 10px;
        border: 1px solid #2d3748;
        font-size: 0.8rem;
        margin-bottom: 10px;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] { background-color: #0e1117; border-right: 1px solid #2d3748; }
</style>
"""
st.markdown(TERMINAL_CSS, unsafe_allow_html=True)

# AI Motor (Lokal - Egitilmis Model)
st.cache_resource.clear()
@st.cache_resource
def load_analyzer():
    return Analyzer()

@st.cache_resource
def load_ai_engine():
    try:
        engine = LiveInferenceEngine()
        if engine.is_ready:
            print("[TERMINAL] AI Motoru HAZIR")
        else:
            print("[TERMINAL] AI Motoru yuklenemedi")
        return engine
    except Exception as e:
        print(f"[TERMINAL] AI Motor hatasi: {e}")
        return None

terminal_analyzer = load_analyzer()
ai_engine = load_ai_engine()

# --- Hisse Listeleri (Master Lists) ---
BIST30 = ["AKBNK", "ARCLK", "ASELS", "ASTOR", "BIMAS", "BRSAN", "EKGYO", "ENKAI", "EREGL", "FROTO", "GARAN", "GUBRF", "HEKTS", "ISCTR", "KCHOL", "KONTR", "KOZAL", "KRDMD", "ODAS", "OYAKC", "PETKM", "PGSUS", "SAHOL", "SASA", "SISE", "TCELL", "THYAO", "TOASO", "TUPRS", "YKBNK"]

BIST100 = sorted(list(set(BIST30 + ["AEFES", "AGHOL", "AKCNS", "AKGRT", "AKSEN", "ALARK", "ALBRK", "ALGYO", "ALKIM", "ARDYZ", "ASUZU", "AYDEM", "BAGFS", "BERA", "BIENY", "BRYAT", "BUCIM", "CANTE", "CCOLA", "CIMSA", "CWENE", "DOAS", "DOHOL", "EGEEN", "ENJSA", "EUPWR", "GENIL", "GESAN", "GLYHO", "GWIND", "HALKB", "IPEKE", "ISDMR", "ISGYO", "ISMEN", "IZMDC", "KARDM", "KAYSE", "KCAER", "KFEIN", "KORDS", "KOZAA", "MAVI", "MGROS", "MIATK", "NETAS", "OTKAR", "PENTA", "QUAGR", "SAYAS", "SDTTR", "SKBNK", "SMRTG", "SOKM", "TABGD", "TAVHL", "TKFEN", "TKNSA", "TMSN", "TSKB", "TTKOM", "TTRAK", "TURSG", "ULKER", "VAKBN", "VESBE", "VESTL", "YEOTK", "ZOREN"])))

TUM_BIST = sorted(list(set(BIST100 + ["A1CAP", "ACSEL", "ADEL", "ADESE", "ADGYO", "AFYON", "AGES", "AGROT", "AGYO", "AHGAZ", "AHSGY", "AKENR", "AKFGY", "AKMGY", "AKSA", "AKSGY", "AKSUE", "AKYHO", "ALCAR", "ALCTL", "ALFAS", "ALKA", "ALMAD", "ALTNY", "ANELE", "ANGEN", "ANHYT", "ANSGR", "ARASE", "ARENA", "ARSAN", "ARZUM", "ASGYO", "ATAKP", "ATATP", "ATEKS", "ATLAS", "ATPSY", "AVGYO", "AVHOL", "AVOD", "AVTUR", "AYCES", "AYDEM", "AYEN", "AYES", "AYGAZ", "AZTEK", "BAKAB", "BALAT", "BANVT", "BARMA", "BASCM", "BASGZ", "BAYRK", "BEGYO", "BERK", "BESLR", "BEYAZ", "BFREN", "BIGCH", "BINBN", "BINHO", "BIOEN", "BIZIM", "BJKAS", "BLCYT", "BMSCH", "BMSTL", "BNTAS", "BOBET", "BORLS", "BOSSA", "BRISA", "BRKO", "BRKSN", "BRKVY", "BRLSM", "BRMEN", "BSOKE", "BTCIM", "BUCIM", "BURCE", "BURVA", "BVSAN", "BYDNR", "CASA", "CATES", "CELHA", "CEMAS", "CEMTS", "CEOEM", "CLEBI", "CMBTN", "CMENT", "CONSE", "COSMO", "CRDFA", "CRFSA", "CUSAN", "CVKMD", "DAGH", "DAPGM", "DARDL", "DAREN", "DENGE", "DERHL", "DERIM", "DESA", "DESPC", "DEVA", "DGATE", "DGGYO", "DGNMO", "DIRIT", "DITAS", "DMSAS", "DNISI", "DOBUR", "DOGUB", "DOKTA", "DOYLE", "DURDO", "DYOBY", "DZGYO", "EBEBK", "ECILC", "ECZYT", "EDATA", "EDIP", "EGGUB", "EGPRO", "EGSER", "EKIZ", "EKSUN", "ELITE", "EMNIS", "ENSRI", "ENTRA", "EPLAS", "ERSU", "ESCAR", "ESCOM", "ESEN", "ETILR", "ETYAT", "EUHOL", "EUREN", "EUYO", "FADE", "FENER", "FLAP", "FMIZP", "FONET", "FORMT", "FORTE", "FRIGO", "FZLGY", "GARFA", "GEDIK", "GEDZA", "GENTS", "GEREL", "GERSAN", "GGLO", "GIPTA", "GLBMD", "GLRYH", "GMTAS", "GOKNR", "GOLTS", "GOODY", "GOZDE", "GPNTP", "GRNYO", "GRSEL", "GSDDE", "GSDHO", "GUNDG", "GZNMI", "HATEK", "HATSN", "HDFGS", "HEDEF", "HKTM", "HLGYO", "HRKET", "HTTBT", "HUBVC", "HUNER", "HURGZ", "ICBCT", "IDEAS", "IDGYO", "IEYHO", "IHEVA", "IHGZT", "IHLAS", "IHLGM", "IHYAY", "IMASM", "INDES", "INFO", "INGRM", "INTEM", "INVEO", "INVES", "IPEKE", "ISATR", "ISBIR", "ISBTR", "ISFIN", "ISGSY", "ISKPL", "ISKUR", "ISSEN", "ISYAT", "IZFAS", "IZMDC", "IZENR", "JANTS", "KAPLM", "KAREL", "KARSN", "KARTN", "KARYE", "KATMR", "KBORU", "KENT", "KERVN", "KERVT", "KGYO", "KILIZ", "KIMMR", "KLGYO", "KLKIM", "KLMSN", "KLNMA", "KLRHO", "KLSYN", "KMPUR", "KNFRT", "KOCMT", "KONKA", "KONYA", "KOPOL", "KOTON", "KRGYO", "KRONT", "KRPLS", "KRSTL", "KRTEK", "KRVGD", "KSTUR", "KTLEV", "KTSKR", "KUTPO", "KUVVA", "KUYAS", "KZBGY", "KZGYO", "LIDER", "LIDFA", "LILAK", "LINK", "LKMNH", "LMKDC", "LOGO", "LUKSK", "MAALT", "MACKO", "MAGEN", "MAKIM", "MAKTK", "MANAS", "MARBL", "MARKA", "MARTI", "MEDTR", "MEGAP", "MEKAG", "MENTD", "MEPET", "MERCN", "MERIT", "MERKO", "METRO", "METUR", "MHRGY", "MIPAZ", "MKRS", "MNDRS", "MOBTL", "MPARK", "MRGYO", "MRSHL", "MSGYO", "MTRKS", "MTRYO", "MZHLD", "NATEN", "NIBAS", "NTGAZ", "NTHOL", "NUGYO", "NUHCM", "OBAMS", "OBAS", "ODINE", "OFSYM", "ONCSM", "ORCAY", "ORGE", "ORMA", "OSMEN", "OSTIM", "OYLUM", "OYOYO", "OZGYO", "OZKGY", "OZRDN", "OZSUB", "PAGYO", "PAMEL", "PARSN", "PASEU", "PATEK", "PCILT", "PEGYO", "PEKGY", "PENGD", "PINSU", "PKART", "PKENT", "PLAT", "PNLSN", "PNSUT", "POLHO", "POLTK", "PRDGS", "PRKAB", "PRKME", "PRZMA", "PSDTC", "PSGYO", "QNBFB", "QUAGR", "RALYH", "RAYSG", "REEDR", "RGYAS", "RNPOL", "RODRG", "ROYAL", "RTALB", "RUBNS", "RYGYO", "RYSAS", "SAFKR", "SAMAT", "SANEL", "SANFM", "SANKO", "SARKY", "SAYAS", "SEGYO", "SEKFK", "SEKUR", "SELEC", "SELGD", "SELVA", "SEYKM", "SILVR", "SKTAS", "SMART", "SNAI", "SNICA", "SNPAM", "SODSN", "SOKE", "SONME", "SRVGY", "SUMAS", "SUNGW", "SURGY", "SUWEN", "TARKM", "TATGD", "TBORG", "TDGYO", "TEKTU", "TERRA", "TGSAS", "TLMAN", "TMPOL", "TNZTP", "TRCAS", "TRGYO", "TRILC", "TSPOR", "TUCLK", "TUKAS", "TUREX", "TURGG", "UFUK", "ULAS", "ULUFA", "ULUSE", "ULUUN", "UMPAS", "UNLU", "USAK", "UZERB", "VAKFN", "VAKKO", "VANGD", "VBTYZ", "VERUS", "VKFYO", "VKGYO", "VKING", "VRGYO", "YAPRK", "YATAS", "YAYLA", "YBTAS", "YEOTK", "YESIL", "YGGYO", "YGYO", "YKSLN", "YONGA", "YUNSA", "YYAPI", "YYLGD", "ZEDUR", "ZRGYO"])))

# --- Core Processing ---
if 'results' not in st.session_state: st.session_state.results = []

def perform_resample(df, rule='4H'):
    try:
        df_copy = df.copy()
        if 'Date' not in df_copy.columns: df_copy['Date'] = df_copy.index
        df_copy.set_index('Date', inplace=True)
        agg_dict = {'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'}
        res = df_copy.resample(rule).agg(agg_dict).dropna()
        res.reset_index(inplace=True)
        return res
    except: return df

@st.cache_data(ttl=300)
def fetch_terminal_data(ticker, interval, period, resample=None):
    try:
        symbol = f"{ticker}.IS"
        df = yf.download(symbol, period=period, interval=interval, progress=False)
        
        if df is None or df.empty or len(df) < 50: return None
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        
        df = df.rename(columns={'Open':'Open', 'High':'High', 'Low':'Low', 'Close':'Close', 'Volume':'Volume'})
        df = df.ffill().dropna()
        
        # Resample (2h, 4h) — cache'lenir, her hisse icin 1 kez
        if resample:
            if 'Date' not in df.columns: df['Date'] = df.index
            df.set_index('Date', inplace=True)
            agg = {'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'}
            df = df.resample(resample).agg(agg).dropna()
            df.reset_index(inplace=True)
            if len(df) < 50: return None
        
        return df
    except Exception as e:
        print(f"[HATA] {ticker}: {e}")
        return None

def get_ai_prediction(hisse, df):
    """Lokal AI motoru ile tahmin yapar."""
    try:
        if ai_engine is None or not ai_engine.is_ready:
            return {"hata": "AI MOTORU YUKLENEMEDI"}
        
        df_copy = df.copy()
        # Kolon isimlerini duzelt
        col_map = {c: c.capitalize() for c in df_copy.columns}
        df_copy = df_copy.rename(columns=col_map)
        
        # Gerekli kolonlari kontrol et
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            if col not in df_copy.columns:
                return {"hata": "VERI FORMATI UYUMSUZ"}
        
        result = ai_engine.predict_from_ohlcv(hisse, df_copy)
        return result
    except Exception as e:
        print(f"[AI HATA] {hisse}: {e}")
        return {"hata": "AI ANALIZ HATASI"}

def calc_change(df, bars):
    try:
        if len(df) < bars + 1: return None
        curr = float(df.iloc[-1]['Close'])
        prev = float(df.iloc[-(bars+1)]['Close'])
        return ((curr - prev) / prev) * 100
    except: return None

def analyze_tech(df, selected_patterns, timeframe_label):
    """Sadece secilen formasyonlari arar. Secilmeyen hicbir formasyon calistirilmaz."""
    res = []
    df_ana = df.copy()
    if 'Date' not in df_ana.columns: df_ana['Date'] = df_ana.index
    
    try:
        df_ind = terminal_analyzer.add_indicators(df_ana)
        found = []
        
        # Her formasyon icin AYRI AYRI calistir
        if "TOBO" in selected_patterns:
            zz = terminal_analyzer.calculate_zigzag(df_ind, deviation=0.04)
            found.extend(terminal_analyzer.detect_tobo_zigzag(df_ind, zz, timeframe_label))
        
        if "OBO" in selected_patterns:
            found.extend(terminal_analyzer.detect_obo_pattern(df_ind, timeframe_label))
        
        if "Fincan Kulp" in selected_patterns:
            zz = terminal_analyzer.calculate_zigzag(df_ind, deviation=0.04)
            found.extend(terminal_analyzer.detect_cup_zigzag(df_ind, zz, timeframe_label))
        
        if "Boğa Bayrağı" in selected_patterns:
            found.extend(terminal_analyzer.detect_flag_pattern(df_ind, timeframe_label) if hasattr(terminal_analyzer, 'detect_flag_pattern') else [])
        
        if "High Tight Flag (Roket)" in selected_patterns:
            found.extend(terminal_analyzer.detect_high_tight_flag(df_ind))
        
        if "RSI Uyumsuzluğu" in selected_patterns:
            zz = terminal_analyzer.calculate_zigzag(df_ind)
            found.extend(terminal_analyzer.detect_rsi_divergence(df_ind, zz, timeframe_label))
        
        if "Mum Formasyonları" in selected_patterns:
            found.extend(terminal_analyzer.detect_candlestick_patterns(df_ind, timeframe_label))
            
        for f in found:
            res.append({
                "Name": f.get('name', 'Bilinmeyen'),
                "Desc": f.get('desc', ''),
                "target": f.get('target', None),
                "stop": f.get('stop', None),
                "score": f.get('score', 0),
                "status": f.get('status', ''),
                "type": f.get('type', ''),
            })
    except Exception as e:
        print(f"[TECH HATA] {e}")
    return res


def create_ticker_chart(ticker, df, tech_results):
    """Her hisse icin mumlu grafik + hacim + formasyon isaretleri olusturur."""
    try:
        df_chart = df.copy()
        if 'Date' not in df_chart.columns:
            df_chart['Date'] = df_chart.index
        df_chart = df_chart.tail(120)  # Son 120 bar
        
        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.75, 0.25]
        )
        
        # Mum grafigi
        fig.add_trace(go.Candlestick(
            x=df_chart['Date'],
            open=df_chart['Open'],
            high=df_chart['High'],
            low=df_chart['Low'],
            close=df_chart['Close'],
            name='Fiyat',
            increasing_line_color='#00ff88',
            decreasing_line_color='#ff4444',
            increasing_fillcolor='#00ff88',
            decreasing_fillcolor='#ff4444',
        ), row=1, col=1)
        
        # SMA 20 ve 50
        if len(df_chart) >= 20:
            sma20 = df_chart['Close'].rolling(20).mean()
            fig.add_trace(go.Scatter(
                x=df_chart['Date'], y=sma20,
                name='SMA 20', line=dict(color='#ffbf00', width=1),
            ), row=1, col=1)
        if len(df_chart) >= 50:
            sma50 = df_chart['Close'].rolling(50).mean()
            fig.add_trace(go.Scatter(
                x=df_chart['Date'], y=sma50,
                name='SMA 50', line=dict(color='#00f2ff', width=1),
            ), row=1, col=1)
        
        # Formasyon isaretleri (son barda marker)
        if tech_results:
            last_date = df_chart['Date'].iloc[-1]
            last_high = df_chart['High'].iloc[-1]
            pattern_names = [t['Name'] for t in tech_results[:3]]
            label_text = " | ".join(pattern_names)
            
            fig.add_trace(go.Scatter(
                x=[last_date],
                y=[last_high * 1.02],
                mode='markers+text',
                marker=dict(color='#00f2ff', size=12, symbol='triangle-down'),
                text=[label_text],
                textposition='top center',
                textfont=dict(color='#00f2ff', size=10),
                name='Formasyon',
                showlegend=False,
            ), row=1, col=1)
        
        # Hacim
        colors = ['#00ff88' if c >= o else '#ff4444' 
                  for c, o in zip(df_chart['Close'], df_chart['Open'])]
        fig.add_trace(go.Bar(
            x=df_chart['Date'],
            y=df_chart['Volume'],
            name='Hacim',
            marker_color=colors,
            opacity=0.5,
        ), row=2, col=1)
        
        fig.update_layout(
            height=350,
            template='plotly_dark',
            paper_bgcolor='#1a1c23',
            plot_bgcolor='#0e1117',
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis_rangeslider_visible=False,
            showlegend=False,
            font=dict(family='Roboto Mono', size=10, color='#94a3b8'),
        )
        fig.update_xaxes(gridcolor='#2d3748', showgrid=True)
        fig.update_yaxes(gridcolor='#2d3748', showgrid=True)
        
        return fig
    except Exception as e:
        print(f"[GRAFIK HATA] {ticker}: {e}")
        return None


# --- Header ---
c_title, c_clock = st.columns([3, 1])
with c_title:
    st.markdown("<div class='brand-header'>BORSANEURON | TERMINAL</div>", unsafe_allow_html=True)
with c_clock:
    st.markdown(f"<div style='text-align:right; color:#00f2ff; font-family:Roboto Mono; font-weight:bold;'>SISTEM SAATI: {get_current_tr_time()}</div>", unsafe_allow_html=True)

# --- Sidebar ---
with st.sidebar:
    st.markdown("<div style='color:#00f2ff; font-weight:bold; margin-bottom:10px;'>KONTROL PANELİ</div>", unsafe_allow_html=True)
    scope = st.selectbox("Tarama Kapsamı", ["BIST 30", "BIST 100", "TUM BIST (500+)", "TAKİP LİSTEM"])
    
    active_tickers = []
    if scope == "BIST 30": active_tickers = BIST30
    elif scope == "BIST 100": active_tickers = BIST100
    elif scope == "TUM BIST (500+)": active_tickers = TUM_BIST
    else:
        watchlist = st.text_area("Hisse Kodları (Virgülle)", "THYAO, ASELS")
        active_tickers = [t.strip().upper() for t in watchlist.split(',') if t.strip()]

    period_sel = st.selectbox("Zaman Periyodu", ["Günlük", "Haftalık", "Aylık", "4 Saatlik", "2 Saatlik", "90 Dakikalık", "1 Saatlik"])
    yf_i, yf_p, label = "1d", "max", "Gunluk"
    resample_rule = None
    
    if period_sel == "1 Saatlik": yf_i, yf_p, label = "1h", "730d", "Saatlik"
    elif period_sel == "90 Dakikalık": yf_i, yf_p, label = "90m", "60d", "90Dakika"
    elif period_sel == "2 Saatlik": yf_i, yf_p, label, resample_rule = "1h", "730d", "2Saatlik", "2h"
    elif period_sel == "4 Saatlik": yf_i, yf_p, label = "4h", "60d", "4Saatlik"
    elif period_sel == "Haftalık": yf_i, yf_p, label = "1wk", "max", "Haftalik"
    elif period_sel == "Aylık": yf_i, yf_p, label = "1mo", "max", "Aylik"
    
    patterns = st.multiselect("Formasyonlar", ["TOBO", "OBO", "Fincan Kulp", "Boğa Bayrağı", "Flama", "High Tight Flag (Roket)", "RSI Uyumsuzluğu", "Mum Formasyonları"], default=["Boğa Bayrağı", "TOBO", "Mum Formasyonları"])
    
    st.markdown("---")
    filter_pattern = st.selectbox("Sonuc Filtresi", ["Tümü", "Sadece Formasyon Bulunanlar"] + patterns, index=0)
    show_charts = st.checkbox("Grafikleri Göster", value=True)
    
    if st.button("TERMINAL TARAMASINI BAŞLAT", type="primary", use_container_width=True):
        st.session_state.results = []
        st.session_state.chart_data = {}
        
        prog_bar = st.progress(0)
        status_box = st.empty()
        curr_step = st.empty()
        stat_counter = st.empty()
        
        # ============================================================
        # 1. GECIS: HIZLI FORMASYON TARAMASI (AI yok)
        # ============================================================
        status_box.markdown("<div class='status-msg'>1. GEÇİŞ | FORMASYON TARAMASI...</div>", unsafe_allow_html=True)
        
        candidates = []  # Formasyon bulunan hisseler
        all_scanned = []  # Tum taranan hisseler (formasyon olmayanlar dahil)
        
        for idx, ticker in enumerate(active_tickers):
            prog_bar.progress((idx+1)/len(active_tickers))
            curr_step.caption(f"Formasyon taranıyor: {ticker} ({idx+1}/{len(active_tickers)})")
            
            try:
                df = fetch_terminal_data(ticker, yf_i, yf_p, resample_rule)
                if df is not None:
                    
                    # Sadece formasyon tara (hızlı)
                    tech = analyze_tech(df, patterns, label)
                    p1h = calc_change(df, 1)
                    p1m = calc_change(df, 22)
                    
                    item = {
                        "ticker": ticker,
                        "price": float(df.iloc[-1]['Close']),
                        "p1h": p1h, "p1m": p1m,
                        "tech": tech,
                        "ai": {"karar": "BEKLEMEDE", "guven_orani": 0, "tetikleyici_nedenler": []}
                    }
                    
                    if tech:  # Formasyon bulundu
                        candidates.append(item)
                        st.session_state.chart_data[ticker] = df
                    
                    all_scanned.append(item)
            except:
                pass
        
        stat_counter.markdown(f"<div style='color:#00ff88; font-size:0.85rem;'>1. GEÇİŞ TAMAM: {len(active_tickers)} hisse tarandı → {len(candidates)} hissede formasyon bulundu</div>", unsafe_allow_html=True)
        
        # ============================================================
        # 2. GECIS: SADECE FORMASYON BULUNANLARA AI ANALIZI
        # ============================================================
        if candidates:
            status_box.markdown(f"<div class='status-msg'>2. GEÇİŞ | AI TEYİDİ ({len(candidates)} hisse)...</div>", unsafe_allow_html=True)
            
            for idx, item in enumerate(candidates):
                ticker = item['ticker']
                prog_bar.progress((idx+1)/len(candidates))
                curr_step.caption(f"AI analiz: {ticker} ({idx+1}/{len(candidates)})")
                
                try:
                    df = st.session_state.chart_data.get(ticker)
                    if df is not None:
                        ai = get_ai_prediction(ticker, df)
                        item['ai'] = ai
                except:
                    item['ai'] = {"hata": "AI ANALIZ HATASI"}
        
        # Sonuclari kaydet: Oncelik formasyon bulunanlara
        st.session_state.results = candidates + [s for s in all_scanned if s not in candidates]
        
        status_box.empty()
        curr_step.empty()
        prog_bar.empty()
        stat_counter.empty()

# chart_data init
if 'chart_data' not in st.session_state:
    st.session_state.chart_data = {}

# --- Content ---
if not st.session_state.results:
    st.markdown("<div style='text-align:center; padding:100px; color:#555;'>VERI BAĞLANTISI BEKLENIYOR | LÜTFEN TARAMAYI BAŞLATIN</div>", unsafe_allow_html=True)
else:
    # Filtre uygula
    display_results = st.session_state.results
    if filter_pattern == "Sadece Formasyon Bulunanlar":
        display_results = [r for r in display_results if r['tech']]
        st.markdown(f"<div style='color:#ffbf00; font-size:0.85rem; margin-bottom:10px;'>FILTRE: Formasyon bulunan {len(display_results)} / {len(st.session_state.results)} hisse</div>", unsafe_allow_html=True)
    elif filter_pattern not in ["Tümü", "Sadece Formasyon Bulunanlar"]:
        # Secilen formasyona gore filtrele
        def has_pattern(item, pat_name):
            for t in item.get('tech', []):
                n = t.get('Name', '').upper()
                if pat_name == "TOBO" and "TOBO" in n: return True
                if pat_name == "OBO" and "OBO" in n and "TOBO" not in n: return True
                if pat_name == "Fincan Kulp" and ("CUP" in n or "FINCAN" in n or "CANAK" in n): return True
                if pat_name == "Boğa Bayrağı" and ("FLAG" in n or "BAYRAK" in n or "BAYRA" in n): return True
                if pat_name == "Flama" and "FLAMA" in n: return True
                if pat_name == "High Tight Flag (Roket)" and ("ROKET" in n or "ROCKET" in n or "HTF" in n or "HIGH TIGHT" in n): return True
                if pat_name == "RSI Uyumsuzluğu" and ("RSI" in n or "DIVERGEN" in n): return True
                if pat_name == "Mum Formasyonları" and any(x in n for x in ["ENGULF", "DOJI", "HAMMER", "YUTAN", "MUM", "BULLISH", "BEARISH", "MORNING", "EVENING"]): return True
            return False
        display_results = [r for r in display_results if has_pattern(r, filter_pattern)]
        st.markdown(f"<div style='color:#ffbf00; font-size:0.85rem; margin-bottom:10px;'>FILTRE: {filter_pattern} — {len(display_results)} hisse bulundu</div>", unsafe_allow_html=True)
    
    # Sonuc sayisi
    st.markdown(f"<div style='color:#94a3b8; font-size:0.8rem; margin-bottom:15px;'>GÖSTERILEN: {len(display_results)} HISSE | TARAMA: {scope} | PERIYOT: {period_sel}</div>", unsafe_allow_html=True)
    
    for item in display_results:
        with st.container():
            st.markdown("<div class='terminal-card'>", unsafe_allow_html=True)
            
            l_col, r_col = st.columns([3, 1])
            with l_col:
                st.markdown(f"<span style='font-size:1.8rem; font-weight:700; color:#00f2ff;'>{item['ticker']}</span>  <span style='color:#94a3b8; font-size:1rem;'>₺{item['price']:.2f}</span>", unsafe_allow_html=True)
                
                # Stats
                v1h = f"%{item['p1h']:.2f}" if item['p1h'] is not None else "VERI YOK"
                v1m = f"%{item['p1m']:.2f}" if item['p1m'] is not None else "VERI YOK"
                st.markdown(f"""
                <span class='metric-label'>1 PERIYOT:</span> <span class='metric-value'>{v1h}</span> | 
                <span class='metric-label'>1 AYLIK:</span> <span class='metric-value'>{v1m}</span>
                """, unsafe_allow_html=True)
            
            with r_col:
                ai = item['ai']
                if 'hata' in ai:
                    st.markdown(f"<div class='metric-label'>AI DURUM</div><div style='color:#ffbf00; font-size:0.8rem;'>{ai['hata']}</div>", unsafe_allow_html=True)
                else:
                    karar = ai.get('karar', 'BELIRSIZ')
                    conf = ai.get('guven_orani', 0)
                    karar_color = '#00ff88' if karar == 'AL' else '#ff4444' if karar == 'SAT' else '#ffbf00'
                    st.markdown(f"""
                    <div class='metric-label'>AI TEYİDİ</div>
                    <div style='display:flex; align-items:center; gap:8px; margin-top:4px;'>
                        <span style='background:{karar_color}; color:#0e1117; font-weight:bold; padding:4px 12px; border-radius:3px; font-size:1rem;'>{karar}</span>
                        <span class='metric-value' style='font-size:1.3rem;'>%{conf*100:.1f}</span>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    nedenler = ai.get('tetikleyici_nedenler', [])
                    if nedenler:
                        neden_html = ''.join([f"<div style='color:#94a3b8; font-size:0.7rem; margin-left:4px;'>- {n}</div>" for n in nedenler[:3]])
                        st.markdown(neden_html, unsafe_allow_html=True)
            
            # Formasyon badge'leri + hedef fiyat
            if item['tech']:
                st.markdown("<div style='margin-top:10px; border-top:1px solid #2d3748; padding-top:10px;'>", unsafe_allow_html=True)
                for t in item['tech']:
                    badge_html = f"<span style='background:#00f2ff; color:#0e1117; font-size:0.75rem; font-weight:bold; padding:2px 8px; border-radius:2px;'>{t['Name']}</span>"
                    desc_html = f"<span style='color:#94a3b8; font-size:0.85rem; margin-left:10px;'>{t['Desc']}</span>"
                    
                    # Hedef fiyat ve stop
                    target_html = ""
                    if t.get('target'):
                        target_val = t['target']
                        curr = item['price']
                        pot_pct = ((target_val - curr) / curr) * 100 if curr > 0 else 0
                        target_html += f" <span style='color:#00ff88; font-size:0.8rem; font-weight:bold; margin-left:8px;'>HEDEF: ₺{target_val:.2f} (%{pot_pct:+.1f})</span>"
                    if t.get('stop'):
                        stop_val = t['stop']
                        target_html += f" <span style='color:#ff4444; font-size:0.75rem; margin-left:8px;'>STOP: ₺{stop_val:.2f}</span>"
                    
                    st.markdown(f"<div>{badge_html} {desc_html}{target_html}</div>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
            
            # Grafik
            if show_charts and item['ticker'] in st.session_state.chart_data:
                chart_fig = create_ticker_chart(
                    item['ticker'],
                    st.session_state.chart_data[item['ticker']],
                    item['tech']
                )
                if chart_fig:
                    st.plotly_chart(chart_fig, use_container_width=True, key=f"chart_{item['ticker']}")
            
            st.markdown("</div>", unsafe_allow_html=True)
    
    # ======================================================================
    # OZET TABLOSU (En Altta)
    # ======================================================================
    st.markdown("---")
    st.markdown("<div class='brand-header' style='font-size:1.2rem; margin-top:20px;'>TARAMA OZET TABLOSU</div>", unsafe_allow_html=True)
    
    table_rows = []
    for item in st.session_state.results:
        ai = item['ai']
        karar = ai.get('karar', '-')
        conf = ai.get('guven_orani', 0)
        nedenler = ai.get('tetikleyici_nedenler', [])
        pattern_names = [t['Name'] for t in item['tech']] if item['tech'] else []
        
        # En yuksek hedef fiyati bul
        best_target = None
        best_stop = None
        for t in (item['tech'] or []):
            if t.get('target') and (best_target is None or t['target'] > best_target):
                best_target = t['target']
            if t.get('stop') and (best_stop is None or t['stop'] > best_stop):
                best_stop = t['stop']
        
        pot_pct = round(((best_target - item['price']) / item['price']) * 100, 1) if best_target and item['price'] > 0 else None
        
        table_rows.append({
            "Hisse": item['ticker'],
            "Fiyat (₺)": round(item['price'], 2),
            "Hedef (₺)": round(best_target, 2) if best_target else None,
            "Potansiyel (%)": pot_pct if pot_pct else None,
            "Stop (₺)": round(best_stop, 2) if best_stop else None,
            "1P (%)":  round(item['p1h'], 2) if item['p1h'] else None,
            "AI": karar,
            "Güven (%)": round(conf * 100, 1) if conf else 0,
            "Formasyonlar": ", ".join(pattern_names) if pattern_names else "-",
        })
    
    df_table = pd.DataFrame(table_rows)
    
    # Renklendirme
    def color_karar(val):
        if val == 'AL': return 'background-color: #00ff88; color: #0e1117; font-weight: bold'
        elif val == 'SAT': return 'background-color: #ff4444; color: white; font-weight: bold'
        return ''
    
    def color_change(val):
        if val is None: return ''
        if val > 0: return 'color: #00ff88'
        elif val < 0: return 'color: #ff4444'
        return ''
    
    try:
        styled = df_table.style.map(color_karar, subset=['AI'])
        styled = styled.map(color_change, subset=['1P (%)'])
    except AttributeError:
        # Eski pandas versiyonları için
        styled = df_table.style.applymap(color_karar, subset=['AI'])
        styled = styled.applymap(color_change, subset=['1P (%)'])
    
    styled = styled.set_properties(**{
        'background-color': '#1a1c23',
        'color': '#e2e8f0',
        'border-color': '#2d3748',
        'font-size': '0.8rem',
    })
    
    st.dataframe(df_table, use_container_width=True, height=min(len(df_table)*38 + 40, 600))
    
    # Formasyon ozeti
    pattern_count = sum(1 for r in st.session_state.results if r['tech'])
    al_count = sum(1 for r in st.session_state.results if r['ai'].get('karar') == 'AL')
    sat_count = sum(1 for r in st.session_state.results if r['ai'].get('karar') == 'SAT')
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Toplam Hisse", len(st.session_state.results))
    c2.metric("Formasyon Bulunan", pattern_count)
    c3.metric("AI: AL", al_count)
    c4.metric("AI: SAT", sat_count)

