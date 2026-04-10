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
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from analyzer import Analyzer

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
    
    .status-msg {
        background: #1a1c23;
        color: #ffbf00;
        padding: 10px;
        border: 1px solid #ffbf00;
        font-size: 0.8rem;
        text-transform: uppercase;
    }
    
    .metric-label { font-size: 0.7rem; color: #94a3b8; text-transform: uppercase; }
    .metric-value { color: #00f2ff; font-weight: bold; font-family: 'Roboto Mono', monospace; }
    
    /* Sidebar */
    [data-testid="stSidebar"] { background-color: #0e1117; border-right: 1px solid #2d3748; }
</style>
"""
st.markdown(TERMINAL_CSS, unsafe_allow_html=True)

# AI Connection Bridge
AI_PREDICT_ENDPOINT = "https://borsaneuron-api.onrender.com/predict"

@st.cache_resource
def load_analyzer():
    return Analyzer()

analyzer_v2 = load_analyzer()

# --- Hardcoded Index Management ---
BIST30 = "AKBNK,ARCLK,ASELS,ASTOR,BIMAS,BRSAN,EKGYO,ENKAI,EREGL,FROTO,GARAN,GUBRF,HEKTS,ISCTR,KCHOL,KONTR,KOZAL,KRDMD,ODAS,OYAKC,PETKM,PGSUS,SAHOL,SASA,SISE,TCELL,THYAO,TOASO,TUPRS,YKBNK".split(',')

BIST100 = sorted(list(set(BIST30 + "AEFES,AGHOL,AKCNS,AKGRT,AKSEN,ALARK,ALBRK,ALGYO,ALKIM,ARDYZ,ASUZU,AYDEM,BAGFS,BERA,BIENY,BRYAT,BUCIM,CANTE,CCOLA,CIMSA,CWENE,DOAS,DOHOL,EGEEN,ENJSA,EUPWR,GENIL,GESAN,GLYHO,GUBRF,GWIND,HALKB,HEKTS,IPEKE,ISDMR,ISGYO,ISMEN,IZMDC,KARDM,KAYSE,KCAER,KFEIN,KONTR,KORDS,KOZAA,KOZAL,KRDMD,MAVI,MGROS,MIATK,NETAS,ODAS,OTKAR,OYAKC,PENTA,PETKM,PGSUS,QUAGR,SAHOL,SASA,SAYAS,SDTTR,SISE,SKBNK,SMRTG,SOKM,TABGD,TAVHL,TCELL,THYAO,TKFEN,TKNSA,TMSN,TOASO,TSKB,TTKOM,TTRAK,TUPRS,TURSG,ULKER,VAKBN,VESBE,VESTL,YEOTK,YKBNK,ZOREN".split(','))))

TUM_BIST = sorted(list(set(BIST100 + "A1CAP,ACSEL,ADEL,ADESE,ADGYO,AFYON,AGES,AGROT,AGYO,AHGAZ,AHSGY,AKENR,AKFGY,AKMGY,AKSA,AKSGY,AKSUE,AKYHO,ALCAR,ALCTL,ALFAS,ALKA,ALMAD,ALTNY,ANELE,ANGEN,ANHYT,ANSGR,ARASE,ARENA,ARSAN,ARZUM,ASGYO,ATAKP,ATATP,ATEKS,ATLAS,ATPSY,AVGYO,AVHOL,AVOD,AVTUR,AYCES,AYEN,AYES,AYGAZ,AZTEK,BAKAB,BALAT,BANVT,BARMA,BASCM,BASGZ,BAYRK,BEGYO,BERK,BESLR,BEYAZ,BFREN,BIGCH,BINBN,BINHO,BIOEN,BIZIM,BJKAS,BLCYT,BMSCH,BMSTL,BNTAS,BOBET,BORLS,BOSSA,BRISA,BRKO,BRKSN,BRKVY,BRLSM,BRMEN,BSOKE,BTCIM,BUCIM,BURCE,BURVA,BVSAN,BYDNR,CASA,CATES,CELHA,CEMAS,CEMTS,CEOEM,CLEBI,CMBTN,CMENT,CONSE,COSMO,CRDFA,CRFSA,CUSAN,CVKMD,CWENE,DAGH,DAPGM,DARDL,DAREN,DENGE,DERHL,DERIM,DESA,DESPC,DEVA,DGATE,DGGYO,DGNMO,DIRIT,DITAS,DMSAS,DNISI,DOBUR,DOGUB,DOKTA,DOYLE,DURDO,DYOBY,DZGYO,EBEBK,ECILC,ECZYT,EDATA,EDIP,EGGUB,EGPRO,EGSER,EKIZ,EKSUN,ELITE,EMNIS,ENSRI,ENTRA,EPLAS,ERSU,ESCAR,ESCOM,ESEN,ETILR,ETYAT,EUHOL,EUREN,EUYO,FADE,FENER,FLAP,FMIZP,FONET,FORMT,FORTE,FRIGO,FZLGY,GARFA,GEDIK,GEDZA,GENTS,GEREL,GERSAN,GGLO,GIPTA,GLBMD,GLRYH,GMTAS,GOKNR,GOLTS,GOODY,GOZDE,GPNTP,GRNYO,GRSEL,GSDDE,GSDHO,GUNDG,GZNMI,HATEK,HATSN,HDFGS,HEDEF,HKTM,HLGYO,HRKET,HTTBT,HUBVC,HUNER,HURGZ,ICBCT,IDEAS,IDGYO,IEYHO,IHEVA,IHGZT,IHLAS,IHLGM,IHYAY,IMASM,INDES,INFO,INGRM,INTEM,INVEO,INVES,IPEKE,ISATR,ISBIR,ISBTR,ISFIN,ISGSY,ISKPL,ISKUR,ISSEN,ISYAT,IZFAS,IZMDC,IZENR,JANTS,KAPLM,KAREL,KARSN,KARTN,KARYE,KATMR,KBORU,KENT,KERVN,KERVT,KGYO,KILIZ,KIMMR,KLGYO,KLKIM,KLMSN,KLNMA,KLRHO,KLSYN,KMPUR,KNFRT,KOCMT,KONKA,KONYA,KOPOL,KOTON,KRGYO,KRONT,KRPLS,KRSTL,KRTEK,KRVGD,KSTUR,KTLEV,KTSKR,KUTPO,KUVVA,KUYAS,KZBGY,KZGYO,LIDER,LIDFA,LILAK,LINK,LKMNH,LMKDC,LOGO,LUKSK,MAALT,MACKO,MAGEN,MAKIM,MAKTK,MANAS,MARBL,MARKA,MARTI,MEDTR,MEGAP,MEKAG,MENTD,MEPET,MERCN,MERIT,MERKO,METRO,METUR,MHRGY,MIPAZ,MKRS,MNDRS,MOBTL,MPARK,MRGYO,MRSHL,MSGYO,MTRKS,MTRYO,MZHLD,NATEN,NIBAS,NTGAZ,NTHOL,NUGYO,NUHCM,OBAMS,OBAS,ODINE,OFSYM,ONCSM,ORCAY,ORGE,ORMA,OSMEN,OSTIM,OYLUM,OYOYO,OZGYO,OZKGY,OZRDN,OZSUB,PAGYO,PAMEL,PARSN,PASEU,PATEK,PCILT,PEGYO,PEKGY,PENGD,PINSU,PKART,PKENT,PLAT,PNLSN,PNSUT,POLHO,POLTK,PRDGS,PRKAB,PRKME,PRZMA,PSDTC,PSGYO,QNBFB,QUAGR,RALYH,RAYSG,REEDR,RGYAS,RNPOL,RODRG,ROYAL,RTALB,RUBNS,RYGYO,RYSAS,SAFKR,SAMAT,SANEL,SANFM,SANKO,SARKY,SAYAS,SEGYO,SEKFK,SEKUR,SELEC,SELGD,SELVA,SEYKM,SILVR,SKTAS,SMART,SNAI,SNICA,SNPAM,SODSN,SOKE,SONME,SRVGY,SUMAS,SUNGW,SURGY,SUWEN,TARKM,TATGD,TBORG,TDGYO,TEKTU,TERRA,TGSAS,TLMAN,TMPOL,TNZTP,TRCAS,TRGYO,TRILC,TSPOR,TUCLK,TUKAS,TUREX,TURGG,UFUK,ULAS,ULUFA,ULUSE,ULUUN,UMPAS,UNLU,USAK,UZERB,VAKFN,VAKKO,VANGD,VBTYZ,VERUS,VKFYO,VKGYO,VKING,VRGYO,YAPRK,YATAS,YAYLA,YBTAS,YEOTK,YESIL,YGGYO,YGYO,YKSLN,YONGA,YUNSA,YYAPI,YYLGD,ZEDUR,ZRGYO".split(','))))

# --- Core Logic ---
if 'results_store' not in st.session_state: st.session_state.results_store = []

@st.cache_data(ttl=300)
def fetch_secured_data(ticker, interval, period):
    try:
        session = requests.Session()
        session.headers.update({'User-Agent': 'Mozilla/5.0'})
        symbol = f"{ticker}.IS"
        df = yf.download(symbol, period=period, interval=interval, progress=False, session=session)
        
        if df is None or df.empty or len(df) < 30: return None
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        
        df = df.rename(columns={'Open':'Open', 'High':'High', 'Low':'Low', 'Close':'Close', 'Volume':'Volume'})
        df = df.ffill().dropna()
        return df
    except: return None

def get_ai_prediction(hisse, df):
    try:
        sub_df = df.tail(100).copy()
        sub_df['date'] = sub_df.index.strftime('%Y-%m-%d')
        veriler = sub_df.rename(columns=lambda x: x.lower()).to_dict(orient='records')
        
        payload = {"hisse": hisse, "veriler": veriler}
        response = requests.post(AI_PREDICT_ENDPOINT, json=payload, timeout=12)
        if response.status_code == 200: return response.json()
        return {"hata": "AI TEYIDI BEKLENIYOR"}
    except: return {"hata": "AI TEYIDI BEKLENIYOR"}

def calc_perf(df, bars):
    try:
        if len(df) < bars + 1: return None
        v_now = float(df.iloc[-1]['Close'])
        v_old = float(df.iloc[-(bars+1)]['Close'])
        return ((v_now - v_old) / v_old) * 100
    except: return None

def analyze_tech(df, selected_patterns, timeframe):
    res = []
    df_ana = df.copy()
    if 'Date' not in df_ana.columns: df_ana['Date'] = df_ana.index
    
    analyzer_v2.config['enabled_patterns'] = {
        'tobo': "TOBO" in selected_patterns,
        'obo': "OBO" in selected_patterns,
        'cup': "Fincan Kulp" in selected_patterns,
        'flag': "Boğa Bayrağı" in selected_patterns,
        'flama': "Flama" in selected_patterns,
    }
    
    try:
        tf_label = "Gunluk" if timeframe == "1d" else "Saatlik"
        df_ind = analyzer_v2.add_indicators(df_ana)
        found = analyzer_v2.detect_classic_patterns(df_ind, timeframe=tf_label)
        
        if "High Tight Flag (Roket)" in selected_patterns:
            found.extend(analyzer_v2.detect_high_tight_flag(df_ind))
        if "RSI Uyumsuzluğu" in selected_patterns:
            zz = analyzer_v2.calculate_zigzag(df_ind)
            found.extend(analyzer_v2.detect_rsi_divergence(df_ind, zz, tf_label))
        if "Mum Formasyonları" in selected_patterns:
            found.extend(analyzer_v2.detect_candlestick_patterns(df_ind, tf_label))
            
        for f in found:
            res.append({"Name": f.get('name', 'Bilinmeyen'), "Desc": f.get('desc', '')})
    except: pass
    return res

# --- UI Header ---
tr_col1, tr_col2 = st.columns([3, 1])
with tr_col1:
    st.markdown("<div class='brand-header'>BORSANEURON | TERMINAL</div>", unsafe_allow_html=True)
with tr_col2:
    st.markdown(f"<div style='text-align:right; color:#00f2ff; font-family:Roboto Mono;'>SISTEM SAATI: {get_current_tr_time()}</div>", unsafe_allow_html=True)

# --- Sidebar Controls ---
with st.sidebar:
    st.markdown("<div style='color:#ffbf00; font-weight:bold; margin-bottom:10px;'>TERMINAL KONTROL</div>", unsafe_allow_html=True)
    scope_opt = st.selectbox("Tarama Kapsamı", ["BIST 30", "BIST 100", "TUM BIST (500+)", "ÖZEL TAKİP LİSTESİ"])
    
    tickers_to_scan = []
    if scope_opt == "BIST 30": tickers_to_scan = BIST30
    elif scope_opt == "BIST 100": tickers_to_scan = BIST100
    elif scope_opt == "TUM BIST (500+)": tickers_to_scan = TUM_BIST
    else:
        watchlist = st.text_area("Hisse Kodları (Virgülle)", "THYAO, ASELS")
        tickers_to_scan = [t.strip().upper() for t in watchlist.split(',') if t.strip()]

    period_opt = st.selectbox("Zaman Periyodu", ["Günlük", "4 Saatlik", "1 Saatlik"])
    if period_opt == "Günlük": yf_i, yf_p = "1d", "2y"
    elif period_opt == "4 Saatlik": yf_i, yf_p = "60m", "730d" # Note: 4h requires 1h processing/resampling or just 1h
    else: yf_i, yf_p = "60m", "730d"
    
    pattern_opt = st.multiselect("Formasyonlar", ["TOBO", "OBO", "Fincan Kulp", "Boğa Bayrağı", "Flama", "High Tight Flag (Roket)", "RSI Uyumsuzluğu", "Mum Formasyonları"], default=["Boğa Bayrağı", "TOBO", "Mum Formasyonları"])
    
    if st.button("TERMINAL TARAMASINI BAŞLAT", type="primary", use_container_width=True):
        st.session_state.results_store = []
        
        status_box = st.empty()
        status_box.markdown("<div class='status-msg'>Terminal Meşgul | Tarama Yapılıyor...</div>", unsafe_allow_html=True)
        
        prog_bar = st.progress(0)
        curr_ticker_text = st.empty()
        
        for idx, ticker in enumerate(tickers_to_scan):
            prog_bar.progress((idx+1)/len(tickers_to_scan))
            curr_ticker_text.caption(f"Taranıyor: {ticker}")
            
            df = fetch_secured_data(ticker, yf_i, yf_p)
            if df is not None:
                # 1. Performance
                perf1h = calc_perf(df, 1) if yf_i == "60m" else calc_perf(df, 1)
                perf1m = calc_perf(df, 22)
                
                # 2. Tech
                tech_hits = analyze_tech(df, pattern_opt, yf_i)
                
                # 3. AI
                ai_data = get_ai_prediction(ticker, df)
                
                st.session_state.results_store.append({
                    "ticker": ticker,
                    "price": float(df.iloc[-1]['Close']),
                    "perf1h": perf1h, "perf1m": perf1m,
                    "tech": tech_hits, "ai": ai_data
                })
        
        prog_bar.empty()
        curr_ticker_text.empty()
        status_box.empty()

# --- Content Rendering ---
if not st.session_state.results_store:
    st.markdown("<div style='text-align:center; padding:100px; color:#555;'>VERI BAĞLANTISI BEKLENIYOR | LÜTFEN TARAMAYI BAŞLATIN</div>", unsafe_allow_html=True)
else:
    for item in st.session_state.results_store:
        with st.container():
            st.markdown("<div class='terminal-card'>", unsafe_allow_html=True)
            
            lt, rt = st.columns([3, 1])
            with lt:
                st.markdown(f"<span style='font-size:1.8rem; font-weight:700; color:#00f2ff;'>{item['ticker']}</span>", unsafe_allow_html=True)
                
                # Stats
                p1h = f"%{item['perf1h']:.2f}" if item['perf1h'] is not None else "VERI YOK"
                p1m = f"%{item['perf1m']:.2f}" if item['perf1m'] is not None else "VERI YOK"
                st.markdown(f"""
                <span class='metric-label'>1 SAATLIK:</span> <span class='metric-value'>{p1h}</span> | 
                <span class='metric-label'>1 AYLIK:</span> <span class='metric-value'>{p1m}</span>
                """, unsafe_allow_html=True)
            
            with rt:
                ai = item['ai']
                if 'hata' in ai:
                    st.markdown(f"<div class='metric-label'>AI DURUM</div><div style='color:#ffbf00; font-size:0.8rem;'>{ai['hata']}</div>", unsafe_allow_html=True)
                else:
                    conf = ai.get('guven_orani', 0)
                    st.markdown(f"<div class='metric-label'>AI TEYIDI</div><div class='metric-value' style='font-size:1.5rem;'>%{conf*100:.1f}</div>", unsafe_allow_html=True)
            
            # Formations
            if item['tech']:
                st.markdown("<div style='margin-top:10px; border-top:1px solid #2d3748; padding-top:10px;'>", unsafe_allow_html=True)
                for t in item['tech']:
                    st.markdown(f"<div><span style='background:#00f2ff; color:#0e1117; font-size:0.7rem; font-weight:bold; padding:2px 6px; border-radius:2px;'>{t['Name']}</span> <span style='color:#94a3b8; font-size:0.8rem; margin-left:10px;'>{t['Desc']}</span></div>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
                
            st.markdown("</div>", unsafe_allow_html=True)
