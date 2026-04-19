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
def fetch_terminal_data(ticker, interval, period):
    try:
        symbol = f"{ticker}.IS"
        df = yf.download(symbol, period=period, interval=interval, progress=False)
        
        if df is None or df.empty or len(df) < 50: return None
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        
        df = df.rename(columns={'Open':'Open', 'High':'High', 'Low':'Low', 'Close':'Close', 'Volume':'Volume'})
        df = df.ffill().dropna()
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
    res = []
    df_ana = df.copy()
    if 'Date' not in df_ana.columns: df_ana['Date'] = df_ana.index
    
    terminal_analyzer.config['enabled_patterns'] = {
        'tobo': "TOBO" in selected_patterns,
        'obo': "OBO" in selected_patterns,
        'cup': "Fincan Kulp" in selected_patterns,
        'flag': "Boğa Bayrağı" in selected_patterns,
        'flama': "Flama" in selected_patterns,
    }
    
    try:
        df_ind = terminal_analyzer.add_indicators(df_ana)
        found = terminal_analyzer.detect_classic_patterns(df_ind, timeframe=timeframe_label)
        
        if "High Tight Flag (Roket)" in selected_patterns:
            found.extend(terminal_analyzer.detect_high_tight_flag(df_ind))
        if "RSI Uyumsuzluğu" in selected_patterns:
            zz = terminal_analyzer.calculate_zigzag(df_ind)
            found.extend(terminal_analyzer.detect_rsi_divergence(df_ind, zz, timeframe_label))
        if "Mum Formasyonları" in selected_patterns:
            found.extend(terminal_analyzer.detect_candlestick_patterns(df_ind, timeframe_label))
            
        for f in found:
            res.append({"Name": f.get('name', 'Bilinmeyen'), "Desc": f.get('desc', '')})
    except: pass
    return res

# --- Header ---
c_title, c_clock = st.columns([3, 1])
with c_title:
    st.markdown("<div class='brand-header'>BORSANEURON | TERMINAL</div>", unsafe_allow_html=True)
with c_clock:
    # Drifting clock in sidebar style
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

    period_sel = st.selectbox("Zaman Periyodu", ["Günlük", "Haftalık", "Aylık", "4 Saatlik", "2 Saatlik", "1 Saatlik"])
    yf_i, yf_p, label = "1d", "max", "Gunluk"
    resample_rule = None
    
    if period_sel == "1 Saatlik": yf_i, yf_p, label = "60m", "730d", "Saatlik"
    elif period_sel == "2 Saatlik": yf_i, yf_p, label, resample_rule = "60m", "730d", "2Saatlik", "2h"
    elif period_sel == "4 Saatlik": yf_i, yf_p, label, resample_rule = "60m", "730d", "4Saatlik", "4h"
    elif period_sel == "Haftalık": yf_i, yf_p, label = "1wk", "max", "Haftalik"
    elif period_sel == "Aylık": yf_i, yf_p, label = "1mo", "max", "Aylik"
    
    patterns = st.multiselect("Formasyonlar", ["TOBO", "OBO", "Fincan Kulp", "Boğa Bayrağı", "Flama", "High Tight Flag (Roket)", "RSI Uyumsuzluğu", "Mum Formasyonları"], default=["Boğa Bayrağı", "TOBO", "Mum Formasyonları"])
    
    if st.button("TERMINAL TARAMASINI BAŞLAT", type="primary", use_container_width=True):
        st.session_state.results = []
        
        prog_bar = st.progress(0)
        status_box = st.empty()
        curr_step = st.empty()
        
        status_box.markdown("<div class='status-msg'>TERMINAL MEŞGUL | ANALİZ YAPILIYOR...</div>", unsafe_allow_html=True)
        
        for idx, ticker in enumerate(active_tickers):
            prog_bar.progress((idx+1)/len(active_tickers))
            curr_step.caption(f"Analiz ediliyor: {ticker}")
            
            try:
                df = fetch_terminal_data(ticker, yf_i, yf_p)
                if df is not None:
                    # 1. Resample if 4H
                    if resample_rule:
                        df = perform_resample(df, resample_rule)
                    
                    # 2. Performance
                    p1h = calc_change(df, 1) # ~1 bar hourly/daily etc
                    p1m = calc_change(df, 22) # ~1 month if daily
                    
                    # 3. AI
                    ai = get_ai_prediction(ticker, df)
                    
                    # 4. Tech
                    tech = analyze_tech(df, patterns, label)
                    
                    st.session_state.results.append({
                        "ticker": ticker,
                        "price": float(df.iloc[-1]['Close']),
                        "p1h": p1h, "p1m": p1m,
                        "tech": tech, "ai": ai
                    })
                else:
                    pass # Silent skip
            except:
                pass # Silent skip
                
        status_box.empty()
        curr_step.empty()
        prog_bar.empty()

# --- Content ---
if not st.session_state.results:
    st.markdown("<div style='text-align:center; padding:100px; color:#555;'>VERI BAĞLANTISI BEKLENIYOR | LÜTFEN TARAMAYI BAŞLATIN</div>", unsafe_allow_html=True)
else:
    for item in st.session_state.results:
        with st.container():
            st.markdown("<div class='terminal-card'>", unsafe_allow_html=True)
            
            l_col, r_col = st.columns([3, 1])
            with l_col:
                st.markdown(f"<span style='font-size:1.8rem; font-weight:700; color:#00f2ff;'>{item['ticker']}</span>", unsafe_allow_html=True)
                
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
                    
                    # Tetikleyici nedenler
                    nedenler = ai.get('tetikleyici_nedenler', [])
                    if nedenler:
                        neden_html = ''.join([f"<div style='color:#94a3b8; font-size:0.7rem; margin-left:4px;'>- {n}</div>" for n in nedenler[:3]])
                        st.markdown(neden_html, unsafe_allow_html=True)
            
            if item['tech']:
                st.markdown("<div style='margin-top:10px; border-top:1px solid #2d3748; padding-top:10px;'>", unsafe_allow_html=True)
                for t in item['tech']:
                    st.markdown(f"<div><span style='background:#00f2ff; color:#0e1117; font-size:0.75rem; font-weight:bold; padding:2px 8px; border-radius:2px;'>{t['Name']}</span> <span style='color:#94a3b8; font-size:0.85rem; margin-left:10px;'>{t['Desc']}</span></div>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
