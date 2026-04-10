"""
Teknik Tarayici + AI Entegrasyonu (ACM 465)
BIST hisselerini secilen formasyonlara gore otomatik tarar ve AI tahmini ile dogrular.
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

# Proje kok dizinini yola ekle
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from analyzer import Analyzer

st.set_page_config(page_title="Tarayici + AI", page_icon="🚀", layout="wide")

# Tema ve Stil
try:
    from src.theme import CSS_STYLE
except ImportError:
    from theme import CSS_STYLE

st.markdown(CSS_STYLE, unsafe_allow_html=True)

@st.cache_resource
def get_analyzer():
    return Analyzer()

analyzer_engine = get_analyzer()

TUM_HISSELER_STR = """
A1CAP, ACSEL, ADEL, ADESE, ADGYO, AEFES, AFYON, AGES, AGHOL, AGROT, AGYO, AHGAZ, AHSGY, AKBNK, AKCNS, AKENR, AKFGY, AKGRT, AKMGY, AKSA, AKSEN, AKSGY, AKSUE, AKYHO, ALARK, ALBRK, ALCAR, ALCTL, ALFAS, ALGYO, ALKA, ALKIM, ALMAD, ALTNY, ANELE, ANGEN, ANHYT, ANSGR, ARASE, ARCLK, ARDYZ, ARENA, ARSAN, ARZUM, ASELS, ASGYO, ASTOR, ASUZU, ATAKP, ATATP, ATEKS, ATLAS, ATPSY, AVGYO, AVHOL, AVOD, AVTUR, AYCES, AYDEM, AYEN, AYES, AYGAZ, AZTEK, BAGFS, BAKAB, BALAT, BANVT, BARMA, BASCM, BASGZ, BAYRK, BEGYO, BERA, BERK, BESLR, BEYAZ, BFREN, BIENY, BIGCH, BIMAS, BINBN, BINHO, BIOEN, BIZIM, BJKAS, BLCYT, BMSCH, BMSTL, BNTAS, BOBET, BORLS, BOSSA, BRISA, BRKO, BRKSN, BRKVY, BRLSM, BRMEN, BRSAN, BRYAT, BSOKE, BTCIM, BUCIM, BURCE, BURVA, BVSAN, BYDNR, CANTE, CASA, CATES, CCOLA, CELHA, CEMAS, CEMTS, CEOEM, CIMSA, CLEBI, CMBTN, CMENT, CONSE, COSMO, CRDFA, CRFSA, CUSAN, CVKMD, CWENE, DAGH, DAPGM, DARDL, DAREN, DENGE, DERHL, DERIM, DESA, DESPC, DEVA, DGATE, DGGYO, DGNMO, DIRIT, DITAS, DMSAS, DNISI, DOAS, DOBUR, DOGUB, DOHOL, DOKTA, DOYLE, DURDO, DYOBY, DZGYO, EBEBK, ECILC, ECZYT, EDATA, EDIP, EGEEN, EGGUB, EGPRO, EGSER, EKGYO, EKIZ, EKSUN, ELITE, EMNIS, ENJSA, ENKAI, ENSRI, ENTRA, EPLAS, EREGL, ERSU, ESCAR, ESCOM, ESEN, ETILR, ETYAT, EUHOL, EUREN, EUYO, FADE, FENER, FLAP, FMIZP, FONET, FORMT, FORTE, FRIGO, FROTO, FZLGY, GARAN, GARFA, GEDIK, GEDZA, GENTS, GEREL, GERSAN, GESAN, GGLO, GIPTA, GLBMD, GLRYH, GLYHO, GMTAS, GOKNR, GOLTS, GOODY, GOZDE, GPNTP, GRNYO, GRSEL, GSDDE, GSDHO, GUBRF, GUNDG, GWIND, GZNMI, HALKB, HATEK, HATSN, HDFGS, HEDEF, HEKTS, HKTM, HLGYO, HRKET, HTTBT, HUBVC, HUNER, HURGZ, ICBCT, IDEAS, IDGYO, IEYHO, IHEVA, IHGZT, IHLAS, IHLGM, IHYAY, IMASM, INDES, INFO, INGRM, INTEM, INVEO, INVES, IPEKE, ISATR, ISBIR, ISBTR, ISCTR, ISDMR, ISFIN, ISGSY, ISGYO, ISKPL, ISKUR, ISMEN, ISSEN, ISYAT, IZFAS, IZMDC, IZENR, JANTS, KAPLM, KAREL, KARSN, KARTN, KARYE, KATMR, KAYSE, KBORU, KCAER, KCHOL, KENT, KERVN, KERVT, KFEIN, KGYO, KILIZ, KIMMR, KLGYO, KLKIM, KLMSN, KLNMA, KLRHO, KLSYN, KMPUR, KNFRT, KOCMT, KONKA, KONTR, KONYA, KOPOL, KORDS, KOTON, KOZAL, KOZAA, KRGYO, KRONT, KRPLS, KRSTL, KRTEK, KRVGD, KSTUR, KTLEV, KTSKR, KUTPO, KUVVA, KUYAS, KZBGY, KZGYO, LIDER, LIDFA, LILAK, LINK, LKMNH, LMKDC, LOGO, LUKSK, MAALT, MACKO, MAGEN, MAKIM, MAKTK, MANAS, MARBL, MARKA, MARTI, MAVI, MEDTR, MEGAP, MEKAG, MENTD, MEPET, MERCN, MERIT, MERKO, METRO, METUR, MGROS, MIATK, MHRGY, MIPAZ, MKRS, MNDRS, MOBTL, MPARK, MRGYO, MRSHL, MSGYO, MTRKS, MTRYO, MZHLD, NATEN, NETAS, NIBAS, NTGAZ, NTHOL, NUGYO, NUHCM, OBAMS, OBAS, ODAS, ODINE, OFSYM, ONCSM, ORCAY, ORGE, ORMA, OSMEN, OSTIM, OTKAR, OYAKC, OYLUM, OYOYO, OZGYO, OZKGY, OZRDN, OZSUB, PAGYO, PAMEL, PARSN, PASEU, PATEK, PCILT, PEGYO, PEKGY, PENGD, PENTA, PETKM, PETUN, PGSUS, PINSU, PKART, PKENT, PLAT, PNLSN, PNSUT, POLHO, POLTK, PRDGS, PRKAB, PRKME, PRZMA, PSDTC, PSGYO, QNBFB, QUAGR, RALYH, RAYSG, REEDR, RGYAS, RNPOL, RODRG, ROYAL, RTALB, RUBNS, RYGYO, RYSAS, SAFKR, SAHOL, SAMAT, SANEL, SANFM, SANKO, SARKY, SASA, SAYAS, SDTTR, SEGYO, SEKFK, SEKUR, SELEC, SELGD, SELVA, SEYKM, SILVR, SISE, SKBNK, SKTAS, SMART, SMRTG, SNAI, SNICA, SNPAM, SODSN, SOKE, SOKM, SONME, SRVGY, SUMAS, SUNGW, SURGY, SUWEN, TABGD, TARKM, TATGD, TAVHL, TBORG, TCELL, TDGYO, TEKTU, TERRA, TGSAS, THYAO, TKFEN, TKNSA, TLMAN, TMPOL, TMSN, TNZTP, TOASO, TRCAS, TRGYO, TRILC, TSKB, TSPOR, TTKOM, TTRAK, TUCLK, TUKAS, TUPRS, TUREX, TURGG, TURSG, UFUK, ULAS, ULKER, ULUFA, ULUSE, ULUUN, UMPAS, UNLU, USAK, UZERB, VAKBN, VAKFN, VAKKO, VANGD, VBTYZ, VERUS, VESBE, VESTL, VKFYO, VKGYO, VKING, VRGYO, YAPRK, YATAS, YAYLA, YBTAS, YEOTK, YESIL, YGGYO, YGYO, YKBNK, YKSLN, YONGA, YUNSA, YYAPI, YYLGD, ZEDUR, ZOREN, ZRGYO
"""

BIST30 = "AKBNK,ARCLK,ASELS,ASTOR,BIMAS,BRSAN,EKGYO,ENKAI,EREGL,FROTO,GARAN,GUBRF,HEKTS,ISCTR,KCHOL,KONTR,KOZAL,KRDMD,ODAS,OYAKC,PETKM,PGSUS,SAHOL,SASA,SISE,TCELL,THYAO,TOASO,TUPRS,YKBNK".split(',')

FORMASYON_MAP = {
    "TOBO (Ters Omuz Bas Omuz)": "tobo",
    "OBO (Omuz Bas Omuz)": "obo",
    "Fincan Kulp": "cup",
    "Boga Bayrak": "flag",
    "Flama": "flama",
    "High Tight Flag": "rocket",
    "RSI Uyumsuzluk": "rsi_div",
    "Mum Formasyonlari": "candle",
}

@st.cache_data(ttl=300)
def veri_getir(hisse, bar_sayisi, interval, period, resample_rule=None):
    try:
        symbol = f"{hisse}.IS" if not hisse.endswith(".IS") else hisse
        df = yf.download(symbol, period=period, interval=interval, progress=False)
        if df.empty or len(df) < 20: return None
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        df = df.rename(columns={'Open':'Open', 'High':'High', 'Low':'Low', 'Close':'Close', 'Volume':'Volume'})
        if resample_rule:
            agg_dict = {'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'}
            df = df.resample(resample_rule).agg(agg_dict).dropna()
        df['SMA20'] = df['Close'].rolling(20).mean()
        df['SMA50'] = df['Close'].rolling(50).mean()
        delta = df['Close'].diff()
        gain = delta.clip(lower=0).rolling(window=14).mean()
        loss = (-1 * delta.clip(upper=0)).rolling(window=14).mean().replace(0, 0.0001)
        df['RSI'] = 100 - (100 / (1 + (gain / loss)))
        return df.tail(bar_sayisi)
    except: return None

def grafik_ciz(df, hisse, veri):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.75, 0.25])
    fig.add_trace(go.Candlestick(
        x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Fiyat',
        increasing_line_color='#4ade80', decreasing_line_color='#f87171'
    ), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA20'], line=dict(color='#fbbf24', width=1.2), name='SMA 20'), row=1, col=1)
    if 'Volume' in df.columns:
        colors = ['rgba(74,222,128,0.5)' if c >= o else 'rgba(248,113,113,0.5)' for c, o in zip(df['Close'], df['Open'])]
        fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name='Hacim', marker_color=colors, showlegend=False), row=2, col=1)
    fig.update_layout(height=520, margin=dict(l=10, r=10, t=30, b=10), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='#94a3b8'))
    fig.update_xaxes(rangeslider_visible=False)
    return fig

def analiz_yap(df, secilen_formasyonlar, tolerans, zaman_etiketi, tek_hisse_modu=False):
    if len(df) < 50: return []
    sonuclar = []
    df_work = df.copy()
    if 'Date' not in df_work.columns: df_work['Date'] = df_work.index
    
    analyzer_engine.config['enabled_patterns'] = {
        'tobo': "TOBO (Ters Omuz Bas Omuz)" in secilen_formasyonlar,
        'obo': "OBO (Omuz Bas Omuz)" in secilen_formasyonlar,
        'cup': "Fincan Kulp" in secilen_formasyonlar,
        'flag': "Boga Bayrak" in secilen_formasyonlar,
        'flama': "Flama" in secilen_formasyonlar,
    }
    
    try:
        tf_map = {"GUNLUK": "Gunluk", "HAFTALIK": "Haftalik", "AYLIK": "Aylik", "1 SAAT": "Saatlik", "2 SAAT": "Saatlik", "4 SAAT": "Saatlik"}
        tf = tf_map.get(zaman_etiketi, "Gunluk")
        df_ind = analyzer_engine.add_indicators(df_work)
        patterns = analyzer_engine.detect_classic_patterns(df_ind, timeframe=tf)
        
        # Ek modulleri tarayiciya gore ekle
        if "RSI Uyumsuzluk" in secilen_formasyonlar:
            zz = analyzer_engine.calculate_zigzag(df_ind)
            patterns.extend(analyzer_engine.detect_rsi_divergence(df_ind, zz, tf))
        if "High Tight Flag" in secilen_formasyonlar:
            patterns.extend(analyzer_engine.detect_high_tight_flag(df_ind))
        if "Mum Formasyonlari" in secilen_formasyonlar:
            patterns.extend(analyzer_engine.detect_candlestick_patterns(df_ind, tf))
            
        for p in patterns:
            curr_price = float(df_ind.iloc[-1]['Close'])
            target = float(p.get('target', curr_price * 1.05))
            potansiyel = ((target - curr_price) / curr_price) * 100
            sonuclar.append({
                "Formasyon": p.get('name', 'Bilinmeyen'),
                "Skor": min(p.get('score', 50), 100),
                "Hedef": target, "Stop": p.get('stop', curr_price * 0.95),
                "Potansiyel": potansiyel, "Fiyat": curr_price,
                "Periyot": zaman_etiketi, "Sinyal": p.get('signal', 'Bullish'),
                "Durum": p.get('status', 'Confirmed'), "Kalite": p.get('quality', 'Normal'),
                "Strateji": p.get('strategy', ''), "Vade": p.get('vade', ''),
            })
    except Exception as e:
        print(f"Analiz Hatasi: {e}")

    if not sonuclar and tek_hisse_modu:
        sonuclar.append({
            "Formasyon": "Genel Teknik Gorunum", "Skor": 50, "Hedef": float(df.iloc[-1]['Close']) * 1.05,
            "Stop": float(df.iloc[-1]['Close']) * 0.95, "Potansiyel": 5.0, "Fiyat": float(df.iloc[-1]['Close']),
            "Periyot": zaman_etiketi, "Sinyal": "Notr", "Durum": "", "Kalite": "", "Strateji": "Formasyon yok.", "Vade": "",
        })
    return sonuclar

# UI Sidebar ve Parametreler
with st.sidebar:
    st.markdown("### ⚙️ Ayarlar")
    zaman_secimi = st.selectbox("Periyot", ["GUNLUK (1D)", "HAFTALIK (1W)", "AYLIK (1M)", "4 SAATLIK (4h)", "2 SAATLIK (2h)", "1 SAATLIK (1h)"])
    if "GUNLUK" in zaman_secimi: yf_int, yf_per, z_etiket, yf_res = "1d", "2y", "GUNLUK", None
    elif "HAFTALIK" in zaman_secimi: yf_int, yf_per, z_etiket, yf_res = "1wk", "5y", "HAFTALIK", None
    elif "AYLIK" in zaman_secimi: yf_int, yf_per, z_etiket, yf_res = "1mo", "max", "AYLIK", None
    elif "4 SAATLIK" in zaman_secimi: yf_int, yf_per, z_etiket, yf_res = "60m", "730d", "4 SAAT", "4h"
    elif "2 SAATLIK" in zaman_secimi: yf_int, yf_per, z_etiket, yf_res = "60m", "730d", "2 SAAT", "2h"
    else: yf_int, yf_per, z_etiket, yf_res = "60m", "730d", "1 SAAT", None

    liste_modu = st.radio("Sektor / Liste", ["BIST 30", "TUM BIST", "TEK HISSE (Sniper)"])
    if "TEK HISSE" in liste_modu:
        hisseler = [st.text_input("Hisse Kodu", "THYAO").upper()]
        tek_hisse_aktif = True
    elif "TUM BIST" in liste_modu:
        hisseler = [h.strip() for h in TUM_HISSELER_STR.replace('\n', '').split(',') if len(h) > 1]
        tek_hisse_aktif = False
    else:
        hisseler = BIST30
        tek_hisse_aktif = False

    secilen_formasyonlar = st.multiselect("Taranacak Formasyonlar", list(FORMASYON_MAP.keys()), default=["TOBO (Ters Omuz Bas Omuz)", "Boga Bayrak"])
    bar_sayisi = st.slider("Grafik Bar Sayisi", 50, 300, 150)
    tolerans = st.slider("Hata Toleransi", 1, 10, 3)
    only_confirmed = st.checkbox("Sadece Onaylanmis Formasyonlar", value=False)
    btn_baslat = st.button("🔍 TARAMAYI BASLAT", type="primary", use_container_width=True)

# Gelen sonuclarin saklandigi session state
if 'scan_results' not in st.session_state: st.session_state.scan_results = []
if 'ai_results' not in st.session_state: st.session_state.ai_results = {}

# ==============================================================================
# ANA TARAMA DONGUSU (INTEGRATED WITH AI API)
# ==============================================================================
if btn_baslat:
    temiz_hisseler = sorted(list(set([h.upper() for h in hisseler if len(h) > 1])))
    bar = st.progress(0)
    bulunanlar = []
    st.session_state.ai_results = {}
    
    for i, hisse in enumerate(temiz_hisseler):
        bar.progress((i + 1) / len(temiz_hisseler))
        df = veri_getir(hisse, bar_sayisi, yf_int, yf_per, yf_res)
        
        if df is not None:
            # --- 1. AI API CAGRISI (Zirhli Yapi) ---
            ai_data = {"karar": "YOK", "guven_orani": 0, "tetikleyici_nedenler": []}
            try:
                # Son 100 barı hazırla
                df_last_100 = df.tail(100).copy()
                df_last_100['date'] = df_last_100.index.astype(str)
                # Kolon isimlerini API'nin beklediği formata (küçük harf) çevir
                api_payload = {
                    "hisse": hisse,
                    "veriler": df_last_100.rename(columns=lambda x: x.lower()).to_dict(orient="records")
                }
                # API İstek (Timeout: 5s)
                response = requests.post("https://borsaneuron.onrender.com/predict", json=api_payload, timeout=5)
                if response.status_code == 200:
                    ai_data = response.json()
            except Exception as e:
                # AI sunucusu kapalıysa veya hata verirse sessizce devam et
                pass
            
            # AI sonucunu sakla
            st.session_state.ai_results[hisse] = ai_data
            
            # --- 2. KLASIK ANALIZ ---
            sonuc_listesi = analiz_yap(df, secilen_formasyonlar, tolerans, z_etiket, tek_hisse_aktif)
            for sonuc in sonuc_listesi:
                if only_confirmed and "unconfirmed" in str(sonuc.get('Durum', '')).lower(): continue
                sonuc['Hisse'] = hisse
                # AI verisini sonuca enjekte et
                sonuc['ai_decision'] = ai_data.get('karar', 'YOK')
                sonuc['ai_confidence'] = ai_data.get('guven_orani', 0)
                sonuc['ai_reasons'] = ai_data.get('tetikleyici_nedenler', [])
                bulunanlar.append(sonuc)
    
    bar.empty()
    st.session_state.scan_results = bulunanlar

# ==============================================================================
# SONUCLARI GOSTERME (STREAMLIT UI)
# ==============================================================================
if st.session_state.scan_results:
    st.success(f"Analiz tamamlandi! {len(st.session_state.scan_results)} formasyon bulundu.")
    
    for idx, veri in enumerate(st.session_state.scan_results):
        hisse = veri['Hisse']
        ai_dec = veri.get('ai_decision', 'YOK')
        ai_conf = veri.get('ai_confidence', 0)
        
        # Baslik rengi ve sinyal
        header_text = f"{hisse} | {veri['Formasyon']} | Skor: {veri['Skor']}"
        
        with st.expander(header_text, expanded=True):
            # --- AI ONAY KUTUSU (UI GEREKSINIMI) ---
            if ai_dec == "AL" and ai_conf > 0.85:
                # KOŞUL 1: AI Onaylı AL
                st.success(f"🚀 **YAPAY ZEKA ONAYLI AL** (Güven: %{ai_conf*100:.1f})")
                if veri.get('ai_reasons'):
                    st.caption("🤖 Tetikleyiciler: " + ", ".join(veri['ai_reasons']))
            elif veri.get('Sinyal') == 'Bullish' and ai_dec != "AL":
                # KOŞUL 2: Sadece Klasik AL
                st.info("📉 Klasik formasyon sinyali mevcut, ancak Yapay Zeka henüz teyit etmedi.")

            # Grafik ve Detaylar
            df_plot = veri_getir(hisse, bar_sayisi, yf_int, yf_per, yf_res)
            if df_plot is not None:
                st.plotly_chart(grafik_ciz(df_plot, hisse, veri), use_container_width=True, key=f"ch_{hisse}_{idx}")
            
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Fiyat", f"{veri['Fiyat']:.2f}")
            c2.metric("Hedef", f"{veri['Hedef']:.2f}", f"%{veri['Potansiyel']:.1f}")
            c3.metric("Stop", f"{veri['Stop']:.2f}")
            c4.metric("Güven (AI)", f"%{ai_conf*100:.1f}")
            
            if veri.get('Strateji'): st.markdown(f"**Strateji:** {veri['Strateji']}")
