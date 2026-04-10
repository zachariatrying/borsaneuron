"""
BORSANEURON | TERMINAL VERSION
Bloomberg Inspired Professional Market Scanner
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
from datetime import datetime

# Proje kok dizini
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from analyzer import Analyzer

# Zaman Ayari
TR_TIMEZONE = pytz.timezone('Europe/Istanbul')

def get_tr_now():
    return datetime.now(TR_TIMEZONE).strftime('%H:%M:%S')

# Sayfa Ayarlari
st.set_page_config(
    page_title="BORSANEURON | TERMINAL",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Unified UI Styles (Bloomberg Terminal Concept - Emoji Free)
TERMINAL_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto+Mono:wght@400;700&display=swap');
    
    .stApp { background-color: #0e1117; font-family: 'Roboto Mono', monospace; color: #e2e8f0; }
    
    /* Terminal Card */
    .terminal-card {
        background-color: #1a1c23;
        border: 1px solid #2d3748;
        padding: 20px;
        margin-bottom: 15px;
        border-radius: 4px;
    }
    
    /* Header & Accents */
    .terminal-header {
        color: #00f2ff;
        font-weight: 700;
        letter-spacing: 2px;
        border-bottom: 1px solid #00f2ff;
        padding-bottom: 5px;
        margin-bottom: 15px;
    }
    
    .cyan-text { color: #00f2ff; }
    .amber-text { color: #f59e0b; }
    
    /* Metric & Labels */
    .metric-label { font-size: 0.8rem; color: #94a3b8; text-transform: uppercase; }
    .metric-value { font-size: 1.5rem; font-weight: 700; color: #e2e8f0; }
    
    /* Progress Bar */
    .stProgress > div > div { background-color: #00f2ff !important; }
    
    /* Indicators */
    .status-badge {
        padding: 3px 10px;
        font-size: 0.75rem;
        font-weight: bold;
        border: 1px solid #00f2ff;
        color: #00f2ff;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] { background-color: #0e1117; border-right: 1px solid #2d3748; }
</style>
"""
st.markdown(TERMINAL_CSS, unsafe_allow_html=True)

# API Endpoint
AI_API_URL = "https://borsaneuron-api.onrender.com/predict"

@st.cache_resource
def get_analyzer():
    return Analyzer()

analyzer_engine = get_analyzer()

TUM_HISSELER_STR = """
A1CAP, ACSEL, ADEL, ADESE, ADGYO, AEFES, AFYON, AGES, AGHOL, AGROT, AGYO, AHGAZ, AHSGY, AKBNK, AKCNS, AKENR, AKFGY, AKGRT, AKMGY, AKSA, AKSEN, AKSGY, AKSUE, AKYHO, ALARK, ALBRK, ALCAR, ALCTL, ALFAS, ALGYO, ALKA, ALKIM, ALMAD, ALTNY, ANELE, ANGEN, ANHYT, ANSGR, ARASE, ARCLK, ARDYZ, ARENA, ARSAN, ARZUM, ASELS, ASGYO, ASTOR, ASUZU, ATAKP, ATATP, ATEKS, ATLAS, ATPSY, AVGYO, AVHOL, AVOD, AVTUR, AYCES, AYDEM, AYEN, AYES, AYGAZ, AZTEK, BAGFS, BAKAB, BALAT, BANVT, BARMA, BASCM, BASGZ, BAYRK, BEGYO, BERA, BERK, BESLR, BEYAZ, BFREN, BIENY, BIGCH, BIMAS, BINBN, BINHO, BIOEN, BIZIM, BJKAS, BLCYT, BMSCH, BMSTL, BNTAS, BOBET, BORLS, BOSSA, BRISA, BRKO, BRKSN, BRKVY, BRLSM, BRMEN, BRSAN, BRYAT, BSOKE, BTCIM, BUCIM, BURCE, BURVA, BVSAN, BYDNR, CANTE, CASA, CATES, CCOLA, CELHA, CEMAS, CEMTS, CEOEM, CIMSA, CLEBI, CMBTN, CMENT, CONSE, COSMO, CRDFA, CRFSA, CUSAN, CVKMD, CWENE, DAGH, DAPGM, DARDL, DAREN, DENGE, DERHL, DERIM, DESA, DESPC, DEVA, DGATE, DGGYO, DGNMO, DIRIT, DITAS, DMSAS, DNISI, DOAS, DOBUR, DOGUB, DOHOL, DOKTA, DOYLE, DURDO, DYOBY, DZGYO, EBEBK, ECILC, ECZYT, EDATA, EDIP, EGEEN, EGGUB, EGPRO, EGSER, EKGYO, EKIZ, EKSUN, ELITE, EMNIS, ENJSA, ENKAI, ENSRI, ENTRA, EPLAS, EREGL, ERSU, ESCAR, ESCOM, ESEN, ETILR, ETYAT, EUHOL, EUREN, EUYO, FADE, FENER, FLAP, FMIZP, FONET, FORMT, FORTE, FRIGO, FROTO, FZLGY, GARAN, GARFA, GEDIK, GEDZA, GENTS, GEREL, GERSAN, GESAN, GGLO, GIPTA, GLBMD, GLRYH, GLYHO, GMTAS, GOKNR, GOLTS, GOODY, GOZDE, GPNTP, GRNYO, GRSEL, GSDDE, GSDHO, GUBRF, GUNDG, GWIND, GZNMI, HALKB, HATEK, HATSN, HDFGS, HEDEF, HEKTS, HKTM, HLGYO, HRKET, HTTBT, HUBVC, HUNER, HURGZ, ICBCT, IDEAS, IDGYO, IEYHO, IHEVA, IHGZT, IHLAS, IHLGM, IHYAY, IMASM, INDES, INFO, INGRM, INTEM, INVEO, INVES, IPEKE, ISATR, ISBIR, ISBTR, ISCTR, ISDMR, ISFIN, ISGSY, ISGYO, ISKPL, ISKUR, ISMEN, ISSEN, ISYAT, IZFAS, IZMDC, IZENR, JANTS, KAPLM, KAREL, KARSN, KARTN, KARYE, KATMR, KAYSE, KBORU, KCAER, KCHOL, KENT, KERVN, KERVT, KFEIN, KGYO, KILIZ, KIMMR, KLGYO, KLKIM, KLMSN, KLNMA, KLRHO, KLSYN, KMPUR, KNFRT, KOCMT, KONKA, KONTR, KONYA, KOPOL, KORDS, KOTON, KOZAL, KOZAA, KRGYO, KRONT, KRPLS, KRSTL, KRTEK, KRVGD, KSTUR, KTLEV, KTSKR, KUTPO, KUVVA, KUYAS, KZBGY, KZGYO, LIDER, LIDFA, LILAK, LINK, LKMNH, LMKDC, LOGO, LUKSK, MAALT, MACKO, MAGEN, MAKIM, MAKTK, MANAS, MARBL, MARKA, MARTI, MAVI, MEDTR, MEGAP, MEKAG, MENTD, MEPET, MERCN, MERIT, MERKO, METRO, METUR, MGROS, MIATK, MHRGY, MIPAZ, MKRS, MNDRS, MOBTL, MPARK, MRGYO, MRSHL, MSGYO, MTRKS, MTRYO, MZHLD, NATEN, NETAS, NIBAS, NTGAZ, NTHOL, NUGYO, NUHCM, OBAMS, OBAS, ODAS, ODINE, OFSYM, ONCSM, ORCAY, ORGE, ORMA, OSMEN, OSTIM, OTKAR, OYAKC, OYLUM, OYOYO, OZGYO, OZKGY, OZRDN, OZSUB, PAGYO, PAMEL, PARSN, PASEU, PATEK, PCILT, PEGYO, PEKGY, PENGD, PENTA, PETKM, PETUN, PGSUS, PINSU, PKART, PKENT, PLAT, PNLSN, PNSUT, POLHO, POLTK, PRDGS, PRKAB, PRKME, PRZMA, PSDTC, PSGYO, QNBFB, QUAGR, RALYH, RAYSG, REEDR, RGYAS, RNPOL, RODRG, ROYAL, RTALB, RUBNS, RYGYO, RYSAS, SAFKR, SAHOL, SAMAT, SANEL, SANFM, SANKO, SARKY, SASA, SAYAS, SDTTR, SEGYO, SEKFK, SEKUR, SELEC, SELGD, SELVA, SEYKM, SILVR, SISE, SKBNK, SKTAS, SMART, SMRTG, SNAI, SNICA, SNPAM, SODSN, SOKE, SOKM, SONME, SRVGY, SUMAS, SUNGW, SURGY, SUWEN, TABGD, TARKM, TATGD, TAVHL, TBORG, TCELL, TDGYO, TEKTU, TERRA, TGSAS, THYAO, TKFEN, TKNSA, TLMAN, TMPOL, TMSN, TNZTP, TOASO, TRCAS, TRGYO, TRILC, TSKB, TSPOR, TTKOM, TTRAK, TUCLK, TUKAS, TUPRS, TUREX, TURGG, TURSG, UFUK, ULAS, ULKER, ULUFA, ULUSE, ULUUN, UMPAS, UNLU, USAK, UZERB, VAKBN, VAKFN, VAKKO, VANGD, VBTYZ, VERUS, VESBE, VESTL, VKFYO, VKGYO, VKING, VRGYO, YAPRK, YATAS, YAYLA, YBTAS, YEOTK, YESIL, YGGYO, YGYO, YKBNK, YKSLN, YONGA, YUNSA, YYAPI, YYLGD, ZEDUR, ZOREN, ZRGYO
"""

BIST30 = "AKBNK,ARCLK,ASELS,ASTOR,BIMAS,BRSAN,EKGYO,ENKAI,EREGL,FROTO,GARAN,GUBRF,HEKTS,ISCTR,KCHOL,KONTR,KOZAL,KRDMD,ODAS,OYAKC,PETKM,PGSUS,SAHOL,SASA,SISE,TCELL,THYAO,TOASO,TUPRS,YKBNK".split(',')

# --- Veri Getirme Modulu ---
@st.cache_data(ttl=600)
def veri_getir(hisse, bar_sayisi, interval, period):
    try:
        session = requests.Session()
        session.headers.update({'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'})
        
        symbol = f"{hisse}.IS" if not hisse.endswith(".IS") else hisse
        df = yf.download(symbol, period=period, interval=interval, progress=False, session=session)
        
        if df is None or df.empty or len(df) < 50:
            return None
            
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        df = df.rename(columns={'Open':'Open', 'High':'High', 'Low':'Low', 'Close':'Close', 'Volume':'Volume'})
        
        # Nan Temizligi
        if df['Close'].isnull().any():
            df = df.ffill().dropna()
            
        return df.tail(bar_sayisi)
    except:
        return None

# --- AI Tahmin Modulu ---
def ai_tahmin_al(hisse, df):
    try:
        df_sub = df.tail(100).copy()
        df_sub['date'] = df_sub.index.strftime('%Y-%m-%d')
        
        # Payload Hazirliği
        veriler = df_sub.rename(columns=lambda x: x.lower()).to_dict(orient='records')
        payload = {"hisse": hisse, "veriler": veriler}
        
        response = requests.post(AI_API_URL, json=payload, timeout=15)
        if response.status_code == 200:
            return response.json()
        return {"hata": "Sistem Hazırlanıyor"}
    except:
        return {"hata": "Sistem Hazırlanıyor"}

# --- Analiz Modulu ---
def analiz_yap(df, secilenlar, zaman_etiketi):
    sonuclar = []
    df_work = df.copy()
    if 'Date' not in df_work.columns: df_work['Date'] = df_work.index
    
    # Analyzer Ayarlari
    analyzer_engine.config['enabled_patterns'] = {
        'tobo': "TOBO (Ters Omuz Baş Omuz)" in secilenlar,
        'obo': "OBO (Omuz Baş Omuz)" in secilenlar,
        'cup': "Fincan Kulp" in secilenlar,
        'flag': "Boğa Bayrağı" in secilenlar,
        'flama': "Flama" in secilenlar,
    }
    
    try:
        tf = "Gunluk" if "GUNLUK" in zaman_etiketi else "Saatlik"
        df_ind = analyzer_engine.add_indicators(df_work)
        patterns = analyzer_engine.detect_classic_patterns(df_ind, timeframe=tf)
        
        if "RSI Uyumsuzluğu" in secilenlar:
            zz = analyzer_engine.calculate_zigzag(df_ind)
            patterns.extend(analyzer_engine.detect_rsi_divergence(df_ind, zz, tf))
        
        if "High Tight Flag (Roket)" in secilenlar:
            patterns.extend(analyzer_engine.detect_high_tight_flag(df_ind))
            
        if "Mum Formasyonları" in secilenlar:
            patterns.extend(analyzer_engine.detect_candlestick_patterns(df_ind, tf))
            
        for p in patterns:
            price = float(df_ind.iloc[-1]['Close'])
            sonuclar.append({
                "Formasyon": p.get('name', 'Bilinmeyen'),
                "Skor": p.get('score', 0),
                "Fiyat": price,
                "Hedef": p.get('target', price * 1.05),
                "Stop": p.get('stop', price * 0.95),
                "Sinyal": p.get('signal', 'Bullish'),
                "Tetikleyiciler": p.get('desc', '')
            })
    except: pass
    return sonuclar

# --- Main Page UI ---
col_head, col_time = st.columns([3, 1])
with col_head:
    st.markdown("<h1 class='cyan-text' style='letter-spacing:3px;'>BORSANEURON | TERMINAL</h1>", unsafe_allow_html=True)
with col_time:
    st.markdown(f"<div style='text-align:right; color:#94a3b8; font-size:0.9rem;'>SISTEM SAATI: {get_tr_now()}</div>", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("<h3 class='cyan-text'>AYARLAR</h3>", unsafe_allow_html=True)
    zaman_dilimi = st.selectbox("Periyot", ["GUNLUK (1D)", "HAFTALIK (1W)", "4 SAATLIK (4h)", "1 SAATLIK (1h)"])
    if "GUNLUK" in zaman_dilimi: yf_int, yf_per, z_etiket = "1d", "2y", "GUNLUK"
    elif "HAFTALIK" in zaman_dilimi: yf_int, yf_per, z_etiket = "1wk", "5y", "HAFTALIK"
    else: yf_int, yf_per, z_etiket = "60m", "730d", "SAATLIK"
    
    ticker_input = st.text_input("Hisse Kodu (Virgulle)", "THYAO, GARAN")
    hisseler = [h.strip().upper() for h in ticker_input.split(',')]
    
    secilen_formasyonlar = st.multiselect("Formasyonlar", [
        "TOBO (Ters Omuz Baş Omuz)", "OBO (Omuz Baş Omuz)", "Fincan Kulp",
        "Boğa Bayrağı", "Flama", "High Tight Flag (Roket)", "RSI Uyumsuzluğu", "Mum Formasyonları"
    ], default=["Boğa Bayrağı", "TOBO (Ters Omuz Baş Omuz)"])
    
    btn_scan = st.button("TARAMAYI CALISTIR", type="primary", use_container_width=True)

# Scan Execution
if btn_scan:
    results = []
    bar = st.progress(0)
    for i, h in enumerate(hisseler):
        bar.progress((i+1)/len(hisseler))
        df = veri_getir(h, 200, yf_int, yf_per)
        if df is not None:
            # AI Inference
            ai_res = ai_tahmin_al(h, df)
            # Technical Analysis
            tech_res = analiz_yap(df, secilen_formasyonlar, z_etiket)
            results.append({"hisse": h, "df": df, "ai": ai_res, "tech": tech_res})
    bar.empty()
    st.session_state.last_results = results

# Display
if 'last_results' in st.session_state:
    for res in st.session_state.last_results:
        with st.container():
            st.markdown(f"<div class='terminal-card'>", unsafe_allow_html=True)
            
            # Card Header
            c_h, c_m = st.columns([3, 1])
            with c_h:
                st.markdown(f"<span class='metric-value'>{res['hisse']}</span> <span style='color:#94a3b8;'>SECTOR ANALYSIS</span>", unsafe_allow_html=True)
            with c_m:
                conf = res['ai'].get('guven_orani', 0)
                hata = res['ai'].get('hata')
                if hata:
                    st.markdown(f"<div class='amber-text' style='text-align:right;'>{hata}</div>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<div style='text-align:right;'><div class='metric-label'>AI TRUST</div><div class='cyan-text' style='font-size:1.5rem;'>%{conf*100:.1f}</div></div>", unsafe_allow_html=True)
            
            # Tech Patterns Info
            if res['tech']:
                for p in res['tech']:
                    st.markdown(f"<span class='status-badge'>{p['Formasyon']}</span> <span style='color:#e2e8f0; font-size:0.8rem;'>{p['Tetikleyiciler']}</span>", unsafe_allow_html=True)
            
            # Metrics
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("GIRIS", f"{res['df'].iloc[-1]['Close']:.2f}")
            m2.metric("HEDEF", f"{res['df'].iloc[-1]['Close']*1.05:.2f}", "5.0%")
            m3.metric("STOP", f"{res['df'].iloc[-1]['Close']*0.95:.2f}", "-5.0%", delta_color="inverse")
            m4.metric("PERIYOT", z_etiket)
            
            st.markdown("</div>", unsafe_allow_html=True)
            st.divider()
