"""
BORSANEURON | PROFESSIONAL FINANCE TERMINAL
High-Performance Market Scanner & Pattern Recognition
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

def get_tr_now():
    return datetime.now(TR_TZ).strftime('%H:%M:%S')

# Sayfa Konfigurasyonu
st.set_page_config(
    page_title="BORSANEURON | TERMINAL",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Unified Terminal CSS
TERMINAL_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto+Mono:wght@400;700&display=swap');
    
    .stApp { background-color: #0e1117; color: #e2e8f0; font-family: 'Roboto Mono', monospace; }
    
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
        font-weight: 700;
        font-size: 1.5rem;
        letter-spacing: 2px;
        border-bottom: 2px solid #00f2ff;
        padding-bottom: 10px;
        margin-bottom: 20px;
    }
    
    .metric-label { font-size: 0.75rem; color: #94a3b8; text-transform: uppercase; }
    .metric-value-cyan { color: #00f2ff; font-weight: bold; font-size: 1.2rem; }
    .metric-value-amber { color: #ffbf00; font-weight: bold; font-size: 1.2rem; }
    
    /* Sidebar */
    [data-testid="stSidebar"] { background-color: #0e1117; border-right: 1px solid #2d3748; }
    
    /* No Emojis */
</style>
"""
st.markdown(TERMINAL_CSS, unsafe_allow_html=True)

# AI Endpoint
RENDER_PREDICT_URL = "https://borsaneuron-api.onrender.com/predict"

@st.cache_resource
def get_analyzer_engine():
    return Analyzer()

analyzer_engine = get_analyzer_engine()

# --- Hisse Listeleri (Hardcoded Master Lists) ---
BIST30 = ["AKBNK", "ARCLK", "ASELS", "ASTOR", "BIMAS", "BRSAN", "EKGYO", "ENKAI", "EREGL", "FROTO", "GARAN", "GUBRF", "HEKTS", "ISCTR", "KCHOL", "KONTR", "KOZAL", "KRDMD", "ODAS", "OYAKC", "PETKM", "PGSUS", "SAHOL", "SASA", "SISE", "TCELL", "THYAO", "TOASO", "TUPRS", "YKBNK"]

BIST100 = sorted(list(set(BIST30 + [
    "AEFES", "AGHOL", "AKCNS", "AKGRT", "AKSEN", "ALARK", "ALBRK", "ALGYO", "ALKIM", "ARDYZ", "ASUZU", "AYDEM", "BAGFS", "BERA", "BIENY", "BRYAT", "BUCIM", "CANTE", "CCOLA", "CIMSA", "CWENE", "DOAS", "DOHOL", "EGEEN", "ENJSA", "EUPWR", "GENIL", "GESAN", "GLYHO", "GWIND", "HALKB", "IPEKE", "ISDMR", "ISGYO", "ISMEN", "IZMDC", "KARDM", "KAYSE", "KCAER", "KFEIN", "KORDS", "KOZAA", "MAVI", "MGROS", "MIATK", "NETAS", "OTKAR", "PENTA", "QUAGR", "SAYAS", "SDTTR", "SKBNK", "SMRTG", "SOKM", "TABGD", "TAVHL", "TKFEN", "TKNSA", "TMSN", "TSKB", "TTKOM", "TTRAK", "TURSG", "ULKER", "VAKBN", "VESBE", "VESTL", "YEOTK", "ZOREN"
])))

TUM_BIST = sorted(list(set(BIST100 + [
    "A1CAP", "ACSEL", "ADEL", "ADESE", "ADGYO", "AFYON", "AGES", "AGROT", "AGYO", "AHGAZ", "AHSGY", "AKENR", "AKFGY", "AKMGY", "AKSA", "AKSGY", "AKSUE", "AKYHO", "ALCAR", "ALCTL", "ALFAS", "ALKA", "ALMAD", "ALTNY", "ANELE", "ANGEN", "ANHYT", "ANSGR", "ARASE", "ARENA", "ARSAN", "ARZUM", "ASGYO", "ATAKP", "ATATP", "ATEKS", "ATLAS", "ATPSY", "AVGYO", "AVHOL", "AVOD", "AVTUR", "AYCES", "AYEN", "AYES", "AYGAZ", "AZTEK", "BAKAB", "BALAT", "BANVT", "BARMA", "BASCM", "BASGZ", "BAYRK", "BEGYO", "BERK", "BESLR", "BEYAZ", "BFREN", "BIGCH", "BINBN", "BINHO", "BIOEN", "BIZIM", "BJKAS", "BLCYT", "BMSCH", "BMSTL", "BNTAS", "BOBET", "BORLS", "BOSSA", "BRISA", "BRKO", "BRKSN", "BRKVY", "BRLSM", "BRMEN", "BSOKE", "BTCIM", "BUCIM", "BURCE", "BURVA", "BVSAN", "BYDNR", "CASA", "CATES", "CELHA", "CEMAS", "CEMTS", "CEOEM", "CLEBI", "CMBTN", "CMENT", "CONSE", "COSMO", "CRDFA", "CRFSA", "CUSAN", "CVKMD", "DAGH", "DAPGM", "DARDL", "DAREN", "DENGE", "DERHL", "DERIM", "DESA", "DESPC", "DEVA", "DGATE", "DGGYO", "DGNMO", "DIRIT", "DITAS", "DMSAS", "DNISI", "DOBUR", "DOGUB", "DOKTA", "DOYLE", "DURDO", "DYOBY", "DZGYO", "EBEBK", "ECILC", "ECZYT", "EDATA", "EDIP", "EGGUB", "EGPRO", "EGSER", "EKIZ", "EKSUN", "ELITE", "EMNIS", "ENSRI", "ENTRA", "EPLAS", "ERSU", "ESCAR", "ESCOM", "ESEN", "ETILR", "ETYAT", "EUHOL", "EUREN", "EUYO", "FADE", "FENER", "FLAP", "FMIZP", "FONET", "FORMT", "FORTE", "FRIGO", "FZLGY", "GARFA", "GEDIK", "GEDZA", "GENTS", "GEREL", "GERSAN", "GGLO", "GIPTA", "GLBMD", "GLRYH", "GMTAS", "GOKNR", "GOLTS", "GOODY", "GOZDE", "GPNTP", "GRNYO", "GRSEL", "GSDDE", "GSDHO", "GUNDG", "GZNMI", "HATEK", "HATSN", "HDFGS", "HEDEF", "HKTM", "HLGYO", "HRKET", "HTTBT", "HUBVC", "HUNER", "HURGZ", "ICBCT", "IDEAS", "IDGYO", "IEYHO", "IHEVA", "IHGZT", "IHLAS", "IHLGM", "IHYAY", "IMASM", "INDES", "INFO", "INGRM", "INTEM", "INVEO", "INVES", "IPEKE", "ISATR", "ISBIR", "ISBTR", "ISFIN", "ISGSY", "ISKPL", "ISKUR", "ISSEN", "ISYAT", "IZFAS", "IZMDC", "IZENR", "JANTS", "KAPLM", "KAREL", "KARSN", "KARTN", "KARYE", "KATMR", "KBORU", "KENT", "KERVN", "KERVT", "KGYO", "KILIZ", "KIMMR", "KLGYO", "KLKIM", "KLMSN", "KLNMA", "KLRHO", "KLSYN", "KMPUR", "KNFRT", "KOCMT", "KONKA", "KONYA", "KOPOL", "KOTON", "KRGYO", "KRONT", "KRPLS", "KRSTL", "KRTEK", "KRVGD", "KSTUR", "KTLEV", "KTSKR", "KUTPO", "KUVVA", "KUYAS", "KZBGY", "KZGYO", "LIDER", "LIDFA", "LILAK", "LINK", "LKMNH", "LMKDC", "LOGO", "LUKSK", "MAALT", "MACKO", "MAGEN", "MAKIM", "MAKTK", "MANAS", "MARBL", "MARKA", "MARTI", "MEDTR", "MEGAP", "MEKAG", "MENTD", "MEPET", "MERCN", "MERIT", "MERKO", "METRO", "METUR", "MHRGY", "MIPAZ", "MKRS", "MNDRS", "MOBTL", "MPARK", "MRGYO", "MRSHL", "MSGYO", "MTRKS", "MTRYO", "MZHLD", "NATEN", "NIBAS", "NTGAZ", "NTHOL", "NUGYO", "NUHCM", "OBAMS", "OBAS", "ODINE", "OFSYM", "ONCSM", "ORCAY", "ORGE", "ORMA", "OSMEN", "OSTIM", "OYLUM", "OYOYO", "OZGYO", "OZKGY", "OZRDN", "OZSUB", "PAGYO", "PAMEL", "PARSN", "PASEU", "PATEK", "PCILT", "PEGYO", "PEKGY", "PENGD", "PINSU", "PKART", "PKENT", "PLAT", "PNLSN", "PNSUT", "POLHO", "POLTK", "PRDGS", "PRKAB", "PRKME", "PRZMA", "PSDTC", "PSGYO", "QNBFB", "QUAGR", "RALYH", "RAYSG", "REEDR", "RGYAS", "RNPOL", "RODRG", "ROYAL", "RTALB", "RUBNS", "RYGYO", "RYSAS", "SAFKR", "SAMAT", "SANEL", "SANFM", "SANKO", "SARKY", "SAYAS", "SEGYO", "SEKFK", "SEKUR", "SELEC", "SELGD", "SELVA", "SEYKM", "SILVR", "SKTAS", "SMART", "SNAI", "SNICA", "SNPAM", "SODSN", "SOKE", "SONME", "SRVGY", "SUMAS", "SUNGW", "SURGY", "SUWEN", "TARKM", "TATGD", "TBORG", "TDGYO", "TEKTU", "TERRA", "TGSAS", "TLMAN", "TMPOL", "TNZTP", "TRCAS", "TRGYO", "TRILC", "TSPOR", "TUCLK", "TUKAS", "TUREX", "TURGG", "UFUK", "ULAS", "ULUFA", "ULUSE", "ULUUN", "UMPAS", "UNLU", "USAK", "UZERB", "VAKFN", "VAKKO", "VANGD", "VBTYZ", "VERUS", "VKFYO", "VKGYO", "VKING", "VRGYO", "YAPRK", "YATAS", "YAYLA", "YBTAS", "YEOTK", "YESIL", "YGGYO", "YGYO", "YKSLN", "YONGA", "YUNSA", "YYAPI", "YYLGD", "ZEDUR", "ZRGYO"
])))

# --- Session Management ---
if 'scan_results' not in st.session_state: st.session_state.scan_results = []
if 'scan_active' not in st.session_state: st.session_state.scan_active = False

# --- Helper Functions ---
@st.cache_data(ttl=300)
def fetch_terminal_data(ticker, interval, period):
    try:
        session = requests.Session()
        session.headers.update({'User-Agent': 'Mozilla/5.0'})
        symbol = f"{ticker}.IS"
        df = yf.download(symbol, period=period, interval=interval, progress=False, session=session)
        
        if df is None or df.empty or len(df) < 50: return None
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        
        # NaN Handling
        df = df.rename(columns={'Open':'Open', 'High':'High', 'Low':'Low', 'Close':'Close', 'Volume':'Volume'})
        df = df.ffill().dropna()
        
        return df
    except:
        return None

def get_ai_terminal_prediction(hisse, df):
    try:
        # 100 bars payload
        sub_df = df.tail(100).copy()
        sub_df['date'] = sub_df.index.strftime('%Y-%m-%d')
        veriler = sub_df.rename(columns=lambda x: x.lower()).to_dict(orient='records')
        
        payload = {"hisse": hisse, "veriler": veriler}
        response = requests.post(RENDER_PREDICT_URL, json=payload, timeout=12)
        
        if response.status_code == 200: return response.json()
        return {"hata": "Yapay Zeka Teyidi Bekleniyor"}
    except:
        return {"hata": "Yapay Zeka Teyidi Bekleniyor"}

def calculate_change(df, period_bars):
    if len(df) < period_bars + 1: return None
    curr = float(df.iloc[-1]['Close'])
    prev = float(df.iloc[-(period_bars+1)]['Close'])
    return ((curr - prev) / prev) * 100

def analiz_motoru(df, patterns):
    # Integrated with analyzer.py
    results = []
    df_work = df.copy()
    if 'Date' not in df_work.columns: df_work['Date'] = df_work.index
    
    analyzer_engine.config['enabled_patterns'] = {
        'tobo': "TOBO" in patterns,
        'obo': "OBO" in patterns,
        'cup': "Fincan Kulp" in patterns,
        'flag': "Boğa Bayrağı" in patterns,
        'flama': "Flama" in patterns,
    }
    
    try:
        full_df = analyzer_engine.add_indicators(df_work)
        detected = analyzer_engine.detect_classic_patterns(full_df)
        
        if "High Tight Flag (Roket)" in patterns:
            detected.extend(analyzer_engine.detect_high_tight_flag(full_df))
        if "RSI Uyumsuzluğu" in patterns:
            zz = analyzer_engine.calculate_zigzag(full_df)
            detected.extend(analyzer_engine.detect_rsi_divergence(full_df, zz))
        if "Mum Formasyonları" in patterns:
            detected.extend(analyzer_engine.detect_candlestick_patterns(full_df))
            
        for d in detected:
            results.append({
                "Name": d.get('name', 'Bilinmeyen'),
                "Signal": d.get('signal', 'Bullish'),
                "Desc": d.get('desc', '')
            })
    except: pass
    return results

# --- Main Interface ---
c_title, c_clock = st.columns([3, 1])
with c_title:
    st.markdown("<div class='brand-header'>BORSANEURON | TERMINAL</div>", unsafe_allow_html=True)
with c_clock:
    # Drift Clock using system time normalized to TR
    st.markdown(f"<div style='text-align:right; color:#00f2ff; font-weight:bold;'>SISTEM SAATI: {get_tr_now()}</div>", unsafe_allow_html=True)

# Sidebar Control
with st.sidebar:
    st.markdown("<div style='color:#ffbf00; font-weight:bold; margin-bottom:10px;'>KAMP ANALIZI</div>", unsafe_allow_html=True)
    scope = st.selectbox("Tarama Kapsamı", ["BIST 30", "BIST 100", "TUM BIST (500+)", "OZEL TAKIP LISTESI"])
    
    target_tickers = []
    if scope == "BIST 30": target_tickers = BIST30
    elif scope == "BIST 100": target_tickers = BIST100
    elif scope == "TUM BIST (500+)": target_tickers = TUM_BIST
    else:
        watchlist_input = st.text_area("Hisse Kodları (Virgülle)", "THYAO, ASELS")
        target_tickers = [t.strip().upper() for t in watchlist_input.split(',') if t.strip()]

    interval = st.selectbox("Analiz Periyodu", ["Gunluk", "Saatlik"])
    yf_int = "1d" if interval == "Gunluk" else "60m"
    yf_per = "2y" if interval == "Gunluk" else "730d"
    
    patterns = st.multiselect("Aktif Formasyonlar", ["TOBO", "OBO", "Fincan Kulp", "Boğa Bayrağı", "Flama", "High Tight Flag (Roket)", "RSI Uyumsuzluğu", "Mum Formasyonları"], default=["Boğa Bayrağı", "TOBO", "Mum Formasyonları"])
    
    if st.button("TERMINAL TARAMASINI BAŞLAT", type="primary", use_container_width=True):
        st.session_state.scan_active = True
        st.session_state.scan_results = []
        
        progress = st.progress(0)
        status_text = st.empty()
        
        for i, ticker in enumerate(target_tickers):
            progress.progress((i+1)/len(target_tickers))
            status_text.caption(f"İşleniyor: {ticker}")
            
            df = fetch_terminal_data(ticker, yf_int, yf_per)
            if df is not None:
                # 1. Performance calc
                h1_change = calculate_change(df, 1) if yf_int == "60m" else calculate_change(df, 7) # Approx 1 day if daily
                m1_change = calculate_change(df, 22) # 1 month
                
                # 2. Tech Analysis
                tech = analiz_motoru(df, patterns)
                
                # 3. AI Inference
                ai = get_ai_terminal_prediction(ticker, df)
                
                st.session_state.scan_results.append({
                    "ticker": ticker,
                    "price": float(df.iloc[-1]['Close']),
                    "h1": h1_change,
                    "m1": m1_change,
                    "tech": tech,
                    "ai": ai
                })
        
        status_text.empty()
        progress.empty()
        st.session_state.scan_active = False

# Results Rendering
if not st.session_state.scan_results:
    if st.session_state.scan_active:
        st.info("Sistem Hazırlanıyor... Lütfen bekleyin.")
    else:
        st.markdown("<div style='text-align:center; padding:100px; color:#555;'>VERI BAĞLANTISI BEKLENIYOR | TARAMAYI BAŞLATIN</div>", unsafe_allow_html=True)
else:
    for item in st.session_state.scan_results:
        with st.container():
            st.markdown("<div class='terminal-card'>", unsafe_allow_html=True)
            
            c1, c2 = st.columns([3, 1])
            with c1:
                st.markdown(f"<span style='font-size:1.8rem; font-weight:bold; color:#00f2ff;'>{item['ticker']}</span>", unsafe_allow_html=True)
                
                # Metrics Top
                h_val = f"%{item['h1']:.2f}" if item['h1'] is not None else "Veri Yok"
                m_val = f"%{item['m1']:.2f}" if item['m1'] is not None else "Veri Yok"
                
                st.markdown(f"""
                <span class='metric-label'>1H DEGISIM:</span> <span class='metric-value-cyan'>{h_val}</span> | 
                <span class='metric-label'>1M DEGISIM:</span> <span class='metric-value-cyan'>{m_val}</span>
                """, unsafe_allow_html=True)
                
            with c2:
                ai = item['ai']
                if 'hata' in ai:
                    st.markdown(f"<div class='metric-label'>AI DURUM</div><div class='metric-value-amber' style='font-size:0.9rem;'>{ai['hata']}</div>", unsafe_allow_html=True)
                else:
                    conf = ai.get('guven_orani', 0)
                    st.markdown(f"<div class='metric-label'>AI ONAYI</div><div class='metric-value-cyan'>%{conf*100:.1f}</div>", unsafe_allow_html=True)
            
            # Formations
            if item['tech']:
                st.markdown("<div style='margin-top:15px; border-top:1px solid #2d3748; padding-top:10px;'>", unsafe_allow_html=True)
                for t in item['tech']:
                    st.markdown(f"<div style='margin-bottom:5px;'><span style='background:#00f2ff; color:#0e1117; padding:2px 8px; font-weight:bold; font-size:0.7rem;'>{t['Name']}</span> <span style='color:#94a3b8; font-size:0.8rem; margin-left:10px;'>{t['Desc']}</span></div>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
