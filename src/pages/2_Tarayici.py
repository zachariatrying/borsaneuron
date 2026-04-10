"""
ACM 465 - BIST AI FULL PRO SCANNER
Bloomberg Terminal Style | Dark Mode | AI Integrated
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

# Proje kok dizini ve moduller
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from analyzer import Analyzer

# Sayfa Yapilandirmasi
st.set_page_config(
    page_title="BIST NEURON | PRO",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Tema ve Stil
try:
    from src.theme import CSS_STYLE
except ImportError:
    from theme import CSS_STYLE

# Bloomberg Terminal Custom CSS
BLOOMBERG_CSS = """
<style>
    .bloomberg-card {
        background: #0a0e17;
        border-left: 4px solid #00ff00;
        padding: 15px;
        margin-bottom: 10px;
        font-family: 'Courier New', Courier, monospace;
    }
    .status-badge {
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.7rem;
        font-weight: bold;
        text-transform: uppercase;
    }
    .status-ok { background: #064e3b; color: #4ade80; }
    .status-error { background: #7f1d1d; color: #f87171; }
    
    .ai-metric-high { color: #00ff00; text-shadow: 0 0 10px #00ff00; }
    .ai-metric-low { color: #888888; }
</style>
"""
st.markdown(CSS_STYLE + BLOOMBERG_CSS, unsafe_allow_html=True)

# API Configuration
RENDER_API_URL = "https://borsaneuron-api.onrender.com/predict"

@st.cache_resource
def get_analyzer():
    return Analyzer()

analyzer_engine = get_analyzer()

TUM_HISSELER_STR = """
A1CAP, ACSEL, ADEL, ADESE, ADGYO, AEFES, AFYON, AGES, AGHOL, AGROT, AGYO, AHGAZ, AHSGY, AKBNK, AKCNS, AKENR, AKFGY, AKGRT, AKMGY, AKSA, AKSEN, AKSGY, AKSUE, AKYHO, ALARK, ALBRK, ALCAR, ALCTL, ALFAS, ALGYO, ALKA, ALKIM, ALMAD, ALTNY, ANELE, ANGEN, ANHYT, ANSGR, ARASE, ARCLK, ARDYZ, ARENA, ARSAN, ARZUM, ASELS, ASGYO, ASTOR, ASUZU, ATAKP, ATATP, ATEKS, ATLAS, ATPSY, AVGYO, AVHOL, AVOD, AVTUR, AYCES, AYDEM, AYEN, AYES, AYGAZ, AZTEK, BAGFS, BAKAB, BALAT, BANVT, BARMA, BASCM, BASGZ, BAYRK, BEGYO, BERA, BERK, BESLR, BEYAZ, BFREN, BIENY, BIGCH, BIMAS, BINBN, BINHO, BIOEN, BIZIM, BJKAS, BLCYT, BMSCH, BMSTL, BNTAS, BOBET, BORLS, BOSSA, BRISA, BRKO, BRKSN, BRKVY, BRLSM, BRMEN, BRSAN, BRYAT, BSOKE, BTCIM, BUCIM, BURCE, BURVA, BVSAN, BYDNR, CANTE, CASA, CATES, CCOLA, CELHA, CEMAS, CEMTS, CEOEM, CIMSA, CLEBI, CMBTN, CMENT, CONSE, COSMO, CRDFA, CRFSA, CUSAN, CVKMD, CWENE, DAGH, DAPGM, DARDL, DAREN, DENGE, DERHL, DERIM, DESA, DESPC, DEVA, DGATE, DGGYO, DGNMO, DIRIT, DITAS, DMSAS, DNISI, DOAS, DOBUR, DOGUB, DOHOL, DOKTA, DOYLE, DURDO, DYOBY, DZGYO, EBEBK, ECILC, ECZYT, EDATA, EDIP, EGEEN, EGGUB, EGPRO, EGSER, EKGYO, EKIZ, EKSUN, ELITE, EMNIS, ENJSA, ENKAI, ENSRI, ENTRA, EPLAS, EREGL, ERSU, ESCAR, ESCOM, ESEN, ETILR, ETYAT, EUHOL, EUREN, EUYO, FADE, FENER, FLAP, FMIZP, FONET, FORMT, FORTE, FRIGO, FROTO, FZLGY, GARAN, GARFA, GEDIK, GEDZA, GENTS, GEREL, GERSAN, GESAN, GGLO, GIPTA, GLBMD, GLRYH, GLYHO, GMTAS, GOKNR, GOLTS, GOODY, GOZDE, GPNTP, GRNYO, GRSEL, GSDDE, GSDHO, GUBRF, GUNDG, GWIND, GZNMI, HALKB, HATEK, HATSN, HDFGS, HEDEF, HEKTS, HKTM, HLGYO, HRKET, HTTBT, HUBVC, HUNER, HURGZ, ICBCT, IDEAS, IDGYO, IEYHO, IHEVA, IHGZT, IHLAS, IHLGM, IHYAY, IMASM, INDES, INFO, INGRM, INTEM, INVEO, INVES, IPEKE, ISATR, ISBIR, ISBTR, ISCTR, ISDMR, ISFIN, ISGSY, ISGYO, ISKPL, ISKUR, ISMEN, ISSEN, ISYAT, IZFAS, IZMDC, IZENR, JANTS, KAPLM, KAREL, KARSN, KARTN, KARYE, KATMR, KAYSE, KBORU, KCAER, KCHOL, KENT, KERVN, KERVT, KFEIN, KGYO, KILIZ, KIMMR, KLGYO, KLKIM, KLMSN, KLNMA, KLRHO, KLSYN, KMPUR, KNFRT, KOCMT, KONKA, KONTR, KONYA, KOPOL, KORDS, KOTON, KOZAL, KOZAA, KRGYO, KRONT, KRPLS, KRSTL, KRTEK, KRVGD, KSTUR, KTLEV, KTSKR, KUTPO, KUVVA, KUYAS, KZBGY, KZGYO, LIDER, LIDFA, LILAK, LINK, LKMNH, LMKDC, LOGO, LUKSK, MAALT, MACKO, MAGEN, MAKIM, MAKTK, MANAS, MARBL, MARKA, MARTI, MAVI, MEDTR, MEGAP, MEKAG, MENTD, MEPET, MERCN, MERIT, MERKO, METRO, METUR, MGROS, MIATK, MHRGY, MIPAZ, MKRS, MNDRS, MOBTL, MPARK, MRGYO, MRSHL, MSGYO, MTRKS, MTRYO, MZHLD, NATEN, NETAS, NIBAS, NTGAZ, NTHOL, NUGYO, NUHCM, OBAMS, OBAS, ODAS, ODINE, OFSYM, ONCSM, ORCAY, ORGE, ORMA, OSMEN, OSTIM, OTKAR, OYAKC, OYLUM, OYOYO, OZGYO, OZKGY, OZRDN, OZSUB, PAGYO, PAMEL, PARSN, PASEU, PATEK, PCILT, PEGYO, PEKGY, PENGD, PENTA, PETKM, PETUN, PGSUS, PINSU, PKART, PKENT, PLAT, PNLSN, PNSUT, POLHO, POLTK, PRDGS, PRKAB, PRKME, PRZMA, PSDTC, PSGYO, QNBFB, QUAGR, RALYH, RAYSG, REEDR, RGYAS, RNPOL, RODRG, ROYAL, RTALB, RUBNS, RYGYO, RYSAS, SAFKR, SAHOL, SAMAT, SANEL, SANFM, SANKO, SARKY, SASA, SAYAS, SDTTR, SEGYO, SEKFK, SEKUR, SELEC, SELGD, SELVA, SEYKM, SILVR, SISE, SKBNK, SKTAS, SMART, SMRTG, SNAI, SNICA, SNPAM, SODSN, SOKE, SOKM, SONME, SRVGY, SUMAS, SUNGW, SURGY, SUWEN, TABGD, TARKM, TATGD, TAVHL, TBORG, TCELL, TDGYO, TEKTU, TERRA, TGSAS, THYAO, TKFEN, TKNSA, TLMAN, TMPOL, TMSN, TNZTP, TOASO, TRCAS, TRGYO, TRILC, TSKB, TSPOR, TTKOM, TTRAK, TUCLK, TUKAS, TUPRS, TUREX, TURGG, TURSG, UFUK, ULAS, ULKER, ULUFA, ULUSE, ULUUN, UMPAS, UNLU, USAK, UZERB, VAKBN, VAKFN, VAKKO, VANGD, VBTYZ, VERUS, VESBE, VESTL, VKFYO, VKGYO, VKING, VRGYO, YAPRK, YATAS, YAYLA, YBTAS, YEOTK, YESIL, YGGYO, YGYO, YKBNK, YKSLN, YONGA, YUNSA, YYAPI, YYLGD, ZEDUR, ZOREN, ZRGYO
"""

BIST30 = "AKBNK,ARCLK,ASELS,ASTOR,BIMAS,BRSAN,EKGYO,ENKAI,EREGL,FROTO,GARAN,GUBRF,HEKTS,ISCTR,KCHOL,KONTR,KOZAL,KRDMD,ODAS,OYAKC,PETKM,PGSUS,SAHOL,SASA,SISE,TCELL,THYAO,TOASO,TUPRS,YKBNK".split(',')

# --- Veri Getirme ve Koruma ---
@st.cache_data(ttl=300)
def veri_getir(hisse, bar_sayisi, interval, period, resample_rule=None):
    try:
        symbol = f"{hisse}.IS" if not hisse.endswith(".IS") else hisse
        df = yf.download(symbol, period=period, interval=interval, progress=False)
        
        # NaN Korumasi
        if df is None or df.empty or len(df) < 50:
            return None
        
        # MultiIndex kontrolu
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        df = df.rename(columns={'Open':'Open', 'High':'High', 'Low':'Low', 'Close':'Close', 'Volume':'Volume'})
        
        # Eksik veri kontrolü (NaN atlama)
        if df['Close'].isnull().any():
            df = df.ffill()
            
        if resample_rule:
            agg_dict = {'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'}
            df = df.resample(resample_rule).agg(agg_dict).dropna()
            
        return df.tail(bar_sayisi)
    except Exception:
        return None

# --- AI API Entegrasyonu ---
def get_ai_prediction(hisse, df):
    """
    Render API uzerinden AI tahmini alir.
    Veri uyumlastirma (Padding/Filling) yapar.
    """
    try:
        # Son 100 bar
        df_sub = df.tail(100).copy()
        df_sub['date'] = df_sub.index.strftime('%Y-%m-%d')
        
        # Kolonlari kucuk harfe cevir (API beklentisi)
        df_api = df_sub.rename(columns=lambda x: x.lower())
        
        payload = {
            "hisse": hisse,
            "veriler": df_api.to_dict(orient="records")
        }
        
        response = requests.post(RENDER_API_URL, json=payload, timeout=8)
        
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 503:
            return {"hata": "Sunucu Uyanıyor..."}
        else:
            return {"hata": f"Hata: {response.status_code}"}
            
    except requests.exceptions.Timeout:
        return {"hata": "Zaman Aşımı (Sunucu Uyanıyor?)"}
    except Exception as e:
        return {"hata": "Bağlantı Hatası"}

# --- Grafik Cizimi ---
def grafik_ciz(df, hisse, veri):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.7, 0.3])
    
    # Candlestick
    fig.add_trace(go.Candlestick(
        x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
        name='Fiyat', increasing_line_color='#00ff00', decreasing_line_color='#ff0000',
        increasing_fillcolor='rgba(0,255,0,0.1)', decreasing_fillcolor='rgba(255,0,0,0.1)'
    ), row=1, col=1)
    
    # Detaylar
    fig.add_hline(y=veri.get('Hedef', 0), line_color="#00ff00", line_width=1, line_dash="dash", row=1, col=1)
    
    # Layout (Bloomberg Style)
    fig.update_layout(
        paper_bgcolor='#0a0e17', plot_bgcolor='#0a0e17',
        height=450, margin=dict(l=0, r=0, t=10, b=0),
        xaxis_rangeslider_visible=False,
        showlegend=False,
        font=dict(color='#cbd5e1', family='Courier New')
    )
    fig.update_xaxes(gridcolor='rgba(255,255,255,0.05)')
    fig.update_yaxes(gridcolor='rgba(255,255,255,0.05)', side='right')
    
    return fig

# --- Sidebar ve Filtreler ---
st.sidebar.markdown("""
<div style='text-align:center; padding-bottom:20px;'>
    <h1 style='color:#00ff00; font-family:Courier; font-size:1.5rem;'>BIST NEURON PRO</h1>
    <span style='color:#555;'>Terminal Version 2.0</span>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    zaman_secimi = st.selectbox("Frequency", ["Daily (1D)", "Weekly (1W)", "Hourly (1h)", "4-Hour (4h)"])
    yf_int, yf_per, z_etiket = "1d", "2y", "GUNLUK"
    if "Weekly" in zaman_secimi: yf_int, yf_per, z_etiket = "1wk", "5y", "HAFTALIK"
    elif "Hourly" in zaman_secimi: yf_int, yf_per, z_etiket = "60m", "730d", "1 SAAT"
    elif "4-Hour" in zaman_secimi: yf_int, yf_per, z_etiket = "60m", "730d", "4 SAAT"
    
    hisse_input = st.text_input("Ticker Search", "THYAO")
    hisseler = [h.strip().upper() for h in hisse_input.split(',')]
    
    st.divider()
    formasyonlar = st.multiselect("Active Patterns", ["Bull Flag", "TOBO", "Cup and Handle", "RSI Div"], default=["Bull Flag"])
    btn_scan = st.button("EXECUTE SCAN", type="primary", use_container_width=True)

# --- Header ---
st.markdown(f"""
<div style='background:#0a0e17; padding:15px; border-bottom:1px solid #1e293b;'>
    <div style='display:flex; justify-content:space-between; align-items:center;'>
        <div style='color:#00ff00; font-family:Courier; font-weight:bold; font-size:1.2rem;'>
            📡 MARKET SCANNER | {zaman_secimi.upper()}
        </div>
        <div style='color:#888; font-size:0.8rem;'>
            SYSTEM TIME: {time.strftime('%H:%M:%S')}
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# --- Main Scan Loop ---
# Init session state if needed
if 'results' not in st.session_state: st.session_state.results = []

if btn_scan:
    st.session_state.results = []
    progress_bar = st.progress(0)
    
    for i, t in enumerate(hisseler):
        progress_bar.progress((i+1)/len(hisseler))
        df = veri_getir(t, 200, yf_int, yf_per)
        
        if df is not None:
            # AI Inference
            ai_res = get_ai_prediction(t, df)
            
            # Classic Analysis Placeholder (Simplified for Pro UI)
            price = float(df.iloc[-1]['Close'])
            classic_res = {
                "Hisse": t,
                "Fiyat": price,
                "Hedef": price * 1.05,
                "Stop": price * 0.95,
                "AI": ai_res
            }
            st.session_state.results.append(classic_res)
    progress_bar.empty()

# --- Display Results ---
for res in st.session_state.results:
    with st.container():
        ai = res.get('AI', {})
        conf = ai.get('guven_orani', 0)
        hata = ai.get('hata')
        
        # AI Status Class
        ai_class = "ai-metric-high" if conf > 0.85 else "ai-metric-low"
        ai_label = f"%{conf*100:.1f}" if not hata else "ERROR"
        
        # Card Header
        st.markdown(f"""
        <div class='bloomberg-card'>
            <div style='display:flex; justify-content:space-between; align-items:flex-end;'>
                <div>
                    <span style='color:#eee; font-size:1.4rem; font-weight:bold;'>{res['Hisse']}</span>
                    <span style='color:#888; font-size:0.9rem; margin-left:10px;'>PRICE: {res['Fiyat']:.2f}</span>
                </div>
                <div style='text-align:right;'>
                    <div class='metric-label'>AI CONFIDENCE</div>
                    <div class='{ai_class}' style='font-size:1.8rem; font-weight:bold;'>{ai_label}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        if hata:
            st.warning(f"📡 AI Sunucu Mesajı: {hata} (Tekrar Deneyin)")
        
        col_chart, col_data = st.columns([7, 3])
        
        with col_chart:
            st.plotly_chart(grafik_ciz(veri_getir(res['Hisse'], 100, yf_int, yf_per), res['Hisse'], res), use_container_width=True, key=f"chart_{res['Hisse']}")
            
        with col_data:
            st.markdown("<div style='background:#111; padding:15px; border-radius:10px;'>", unsafe_allow_html=True)
            st.metric("ENTRY", f"{res['Fiyat']:.2f}")
            st.metric("TARGET", f"{res['Hedef']:.2f}", f"+5.0%")
            st.metric("STOP", f"{res['Stop']:.2f}", f"-5.0%", delta_color="inverse")
            
            if ai.get('tetikleyici_nedenler'):
                st.markdown("**SIGNALS:**")
                for n in ai['tetikleyici_nedenler']:
                    st.caption(f"✓ {n}")
            st.markdown("</div>", unsafe_allow_html=True)
            
    st.divider()

if not st.session_state.results and not btn_scan:
    st.info("Market data connection idle. Execute scan to initiate neural analysis.")
