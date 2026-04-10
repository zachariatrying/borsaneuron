"""
================================================================================
ACM 465 - CANLI TAHMIN MOTORU (Live Inference Engine)
live_inference_engine.py
================================================================================
Bu modul, egitilmis ML modelini canli/anlik borsa verisi uzerinde
tahmin yapmak icin kullanir. Herhangi bir projeye import edilebilir
veya FastAPI sunucu olarak calistirilabilir.

Kullanim:
  1. Import:  from live_inference_engine import LiveInferenceEngine
  2. API:     python -m uvicorn live_inference_engine:app --reload --port 8000

API Endpoint:
  POST /predict  ->  {"hisse": "THYAO", "karar": "AL", "guven_orani": 0.88, ...}
================================================================================
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import joblib
from typing import Optional, List, Dict, Any
from datetime import datetime

warnings.filterwarnings('ignore')

# ==============================================================================
# YAPILANDIRMA
# ==============================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Varsayilan dosya yollari — istege gore degistirilebilir
DEFAULT_MODEL_PATH    = os.path.join(BASE_DIR, 'best_model_acm465.joblib')
DEFAULT_SCALER_PATH   = os.path.join(BASE_DIR, 'best_scaler_acm465.joblib')
DEFAULT_FEATURES_PATH = os.path.join(BASE_DIR, 'best_features_acm465.joblib')

# Egitimde kullanilan tum ozellikler (multicollinearity oncesi tam liste)
ALL_FEATURES = [
    'RSI', 'MACD', 'MACD_Signal', 'MACD_Hist',
    'BB_Upper', 'BB_Mid', 'BB_Lower', 'BB_Width',
    'SMA_20', 'SMA_50',
    'Daily_Return', 'Volatility_20',
    'Volume_Ratio',
    'is_bull_flag'
]


# ==============================================================================
# CANLI TAHMIN MOTORU (LIVE INFERENCE ENGINE)
# ==============================================================================

class LiveInferenceEngine:
    """
    Egitilmis BIST ML modelini canli veri uzerinde calistiran tahmin motoru.

    Kullanim:
        engine = LiveInferenceEngine()
        engine = LiveInferenceEngine(model_path="custom_model.joblib")

        result = engine.predict_from_ohlcv("THYAO", df_with_ohlcv_columns)
        # -> {"hisse": "THYAO", "karar": "AL", "guven_orani": 0.88, ...}
    """

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        scaler_path: str = DEFAULT_SCALER_PATH,
        features_path: str = DEFAULT_FEATURES_PATH,
    ):
        """
        Model, Scaler ve ozellik listesini bellege yukler.

        Args:
            model_path:    Egitilmis model dosyasi (.joblib veya .h5)
            scaler_path:   StandardScaler dosyasi (.joblib)
            features_path: Aktif ozellik listesi (.joblib)
        """
        self.model = None
        self.scaler = None
        self.feature_cols = None
        self.model_type = "unknown"
        self._ready = False

        # --- Model Yukleme ---
        try:
            if model_path.endswith('.h5'):
                # Keras modeli
                try:
                    from tensorflow.keras.models import load_model
                    self.model = load_model(model_path)
                    self.model_type = "keras"
                except ImportError:
                    raise ImportError(
                        "Keras/TensorFlow yuklu degil. "
                        ".h5 modeli yuklemek icin 'pip install tensorflow' gerekli."
                    )
            else:
                # Sklearn / joblib modeli
                self.model = joblib.load(model_path)
                self.model_type = "sklearn"
            print(f"[LiveEngine] Model yuklendi: {os.path.basename(model_path)} ({self.model_type})")
        except Exception as e:
            print(f"[LiveEngine] HATA - Model yuklenemedi: {e}")
            return

        # --- Scaler Yukleme ---
        try:
            if os.path.exists(scaler_path):
                self.scaler = joblib.load(scaler_path)
                print(f"[LiveEngine] Scaler yuklendi: {os.path.basename(scaler_path)}")
            else:
                print(f"[LiveEngine] UYARI - Scaler dosyasi bulunamadi: {scaler_path}")
                print(f"[LiveEngine]   -> Yeni StandardScaler olusturulacak (dikkat: uyumsuzluk riski!)")
                from sklearn.preprocessing import StandardScaler
                self.scaler = StandardScaler()
        except Exception as e:
            print(f"[LiveEngine] HATA - Scaler yuklenemedi: {e}")

        # --- Ozellik Listesi Yukleme ---
        try:
            if os.path.exists(features_path):
                self.feature_cols = joblib.load(features_path)
                print(f"[LiveEngine] Ozellik listesi: {self.feature_cols}")
            else:
                # Varsayilan tam listeyi kullan
                self.feature_cols = ALL_FEATURES
                print(f"[LiveEngine] UYARI - Ozellik dosyasi bulunamadi, tam liste kullaniliyor ({len(ALL_FEATURES)} ozellik)")
        except Exception as e:
            self.feature_cols = ALL_FEATURES
            print(f"[LiveEngine] Ozellik yukleme hatasi: {e}")

        self._ready = self.model is not None and self.scaler is not None
        if self._ready:
            print(f"[LiveEngine] Motor HAZIR. Ozellik sayisi: {len(self.feature_cols)}")
        else:
            print(f"[LiveEngine] Motor HAZIR DEGIL!")

    @property
    def is_ready(self) -> bool:
        """Motorun tahmine hazir olup olmadigini dondurur."""
        return self._ready

    # ==========================================================================
    # CANLI OZELLIK MUHENDISLIGI
    # ==========================================================================

    def prepare_live_data(self, raw_data: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        Ham OHLCV verisini modelin anlayacagi formata donusturur.

        Gereken kolonlar: Open, High, Low, Close, Volume (en az 50 satir)
        Hesaplanan ozellikler: RSI, MACD, Bollinger, Bull Flag, vs.

        Args:
            raw_data: DataFrame — Open, High, Low, Close, Volume kolonlari

        Returns:
            pd.DataFrame: Islenilmis ve olceklenmis ozellikler veya None
        """
        if not self._ready:
            print("[LiveEngine] Motor hazir degil!")
            return None

        try:
            df = raw_data.copy()

            # --- Kolon kontrolu ---
            required = ['Open', 'High', 'Low', 'Close', 'Volume']
            missing = [c for c in required if c not in df.columns]
            if missing:
                print(f"[LiveEngine] Eksik kolonlar: {missing}")
                return None

            if len(df) < 50:
                print(f"[LiveEngine] Yetersiz veri: {len(df)} satir (min 50 gerekli)")
                return None

            # --- RSI (14 gun) ---
            delta = df['Close'].diff()
            gain = delta.clip(lower=0).rolling(14).mean()
            loss = (-delta.clip(upper=0)).rolling(14).mean()
            loss = loss.replace(0, 1e-10)
            df['RSI'] = 100 - (100 / (1 + gain / loss))

            # --- MACD ---
            ema12 = df['Close'].ewm(span=12, adjust=False).mean()
            ema26 = df['Close'].ewm(span=26, adjust=False).mean()
            df['MACD'] = ema12 - ema26
            df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
            df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']

            # --- Bollinger Bantlari (20 gun) ---
            df['BB_Mid'] = df['Close'].rolling(20).mean()
            bb_std = df['Close'].rolling(20).std()
            df['BB_Upper'] = df['BB_Mid'] + 2 * bb_std
            df['BB_Lower'] = df['BB_Mid'] - 2 * bb_std
            df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Mid'].replace(0, 1e-10)

            # --- SMA ---
            df['SMA_20'] = df['Close'].rolling(20).mean()
            df['SMA_50'] = df['Close'].rolling(50).mean()

            # --- Getiri ve Volatilite ---
            df['Daily_Return'] = df['Close'].pct_change()
            df['Volatility_20'] = df['Daily_Return'].rolling(20).std()

            # --- Hacim Orani ---
            vol_sma = df['Volume'].rolling(20).mean().replace(0, 1e-10)
            df['Volume_Ratio'] = df['Volume'] / vol_sma

            # --- Boga Bayragi (Bull Flag) ---
            close = df['Close'].values
            high = df['High'].values
            sma20 = df['SMA_20'].values
            flag_signal = np.zeros(len(df))

            for i in range(40, len(df)):
                wh = high[i-40:i].max()
                wl = close[i-40:i].min()
                if wl <= 0:
                    continue
                if (wh - wl) / wl < 0.05:
                    continue
                if close[i] > wh * 0.88 and not np.isnan(sma20[i]) and close[i] > sma20[i]:
                    flag_signal[i] = 1

            df['is_bull_flag'] = flag_signal

            return df

        except Exception as e:
            print(f"[LiveEngine] Ozellik muhendisligi hatasi: {e}")
            return None

    # ==========================================================================
    # TAHMIN
    # ==========================================================================

    def predict_action(self, processed_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Islenilmis veriyi modele sokar ve tahmin + olasilik dondurur.

        Args:
            processed_data: prepare_live_data() ciktisi

        Returns:
            dict: {"karar": "AL"/"SAT", "guven_orani": 0.88, "features_used": [...]}
        """
        if not self._ready:
            return {"hata": "Motor hazir degil"}

        try:
            # Son satiri al (en guncel veri)
            last_row = processed_data.iloc[[-1]]

            # Sadece modelin bekledigini sec
            available = [c for c in self.feature_cols if c in last_row.columns]
            missing_feats = [c for c in self.feature_cols if c not in last_row.columns]

            if missing_feats:
                print(f"[LiveEngine] UYARI - Eksik ozellikler: {missing_feats}")
                # Eksik ozellikleri 0 ile doldur
                for mf in missing_feats:
                    last_row[mf] = 0
                available = self.feature_cols

            X = last_row[self.feature_cols].values
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

            # Olceklendir
            X_scaled = self.scaler.transform(X)

            # Tahmin
            if self.model_type == "keras":
                prob = float(self.model.predict(X_scaled, verbose=0).flatten()[0])
                karar = 1 if prob >= 0.5 else 0
            else:
                karar = int(self.model.predict(X_scaled)[0])
                if hasattr(self.model, 'predict_proba'):
                    prob = float(self.model.predict_proba(X_scaled)[0][1])
                else:
                    prob = float(karar)

            return {
                "karar_kod": karar,
                "karar": "AL" if karar == 1 else "SAT",
                "guven_orani": round(prob, 4),
                "features_used": self.feature_cols,
            }

        except Exception as e:
            return {"hata": str(e)}

    # ==========================================================================
    # TEK ADIMDA TAHMIN (Convenience Method)
    # ==========================================================================

    def predict_from_ohlcv(self, ticker: str, raw_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Ham OHLCV verisinden tek adimda tahmin yapar.

        Args:
            ticker:    Hisse sembolü ("THYAO")
            raw_data:  DataFrame — Open, High, Low, Close, Volume

        Returns:
            dict: Tam tahmin sonucu JSON formati
        """
        # 1. Ozellikleri hesapla
        processed = self.prepare_live_data(raw_data)
        if processed is None:
            return {
                "hisse": ticker,
                "karar": "BELIRSIZ",
                "guven_orani": 0.0,
                "hata": "Veri hazirlanamadi",
                "tetikleyici_nedenler": []
            }

        # 2. Tahmin yap
        result = self.predict_action(processed)

        if "hata" in result:
            return {
                "hisse": ticker,
                "karar": "BELIRSIZ",
                "guven_orani": 0.0,
                "hata": result["hata"],
                "tetikleyici_nedenler": []
            }

        # 3. Tetikleyici nedenleri olustur
        last = processed.iloc[-1]
        nedenler = self._generate_reasons(last, result["karar_kod"])

        return {
            "hisse": ticker,
            "karar": result["karar"],
            "guven_orani": result["guven_orani"],
            "tetikleyici_nedenler": nedenler,
            "detay": {
                "fiyat": round(float(last['Close']), 2),
                "rsi": round(float(last.get('RSI', 0)), 2),
                "macd": round(float(last.get('MACD', 0)), 4),
                "bollinger_width": round(float(last.get('BB_Width', 0)), 4),
                "is_bull_flag": int(last.get('is_bull_flag', 0)),
                "daily_return": round(float(last.get('Daily_Return', 0)), 4),
                "volume_ratio": round(float(last.get('Volume_Ratio', 0)), 2),
            },
            "model_tipi": self.model_type,
            "zaman": datetime.now().isoformat()
        }

    def _generate_reasons(self, row: pd.Series, karar: int) -> List[str]:
        """Tahmin icin insan-okunabilir tetikleyici nedenler uretir."""
        reasons = []
        try:
            rsi = float(row.get('RSI', 50))
            macd = float(row.get('MACD', 0))
            macd_hist = float(row.get('MACD_Hist', 0))
            bull_flag = int(row.get('is_bull_flag', 0))
            vol_ratio = float(row.get('Volume_Ratio', 1))
            bb_width = float(row.get('BB_Width', 0))
            daily_ret = float(row.get('Daily_Return', 0))

            if karar == 1:  # AL
                if rsi < 30:
                    reasons.append("RSI asiri satim bolgesinde (<30)")
                elif rsi < 45:
                    reasons.append("RSI asiri satim bolgesinden cikti")
                if macd_hist > 0:
                    reasons.append("MACD histogram pozitife dondu")
                if bull_flag == 1:
                    reasons.append("Bull Flag Formasyonu onaylandi")
                if vol_ratio > 1.5:
                    reasons.append(f"Hacim ortalamanin {vol_ratio:.1f}x uzerinde")
                if daily_ret > 0.02:
                    reasons.append("Gunluk getiri guclu (+%2 uzerinde)")
                if bb_width < 0.1:
                    reasons.append("Bollinger daralmasi — kırılım bekleniyor")
            else:  # SAT
                if rsi > 70:
                    reasons.append("RSI asiri alim bolgesinde (>70)")
                if macd_hist < 0:
                    reasons.append("MACD histogram negatife dondu")
                if vol_ratio < 0.5:
                    reasons.append("Hacim ortalamanin altinda — ilgi dusuk")
                if daily_ret < -0.02:
                    reasons.append("Gunluk getiri zayif (-%2 altinda)")

            if not reasons:
                reasons.append("Genel teknik gorunum degerlendirmesi")

        except Exception:
            reasons.append("Neden analizi yapilamadi")

        return reasons

    # ==========================================================================
    # TOPLU TAHMIN
    # ==========================================================================

    def predict_batch(self, tickers_data: Dict[str, pd.DataFrame]) -> List[Dict[str, Any]]:
        """
        Birden fazla hisse icin toplu tahmin yapar.

        Args:
            tickers_data: {"THYAO": df_thyao, "GARAN": df_garan, ...}

        Returns:
            List[dict]: Her hisse icin tahmin sonucu listesi
        """
        results = []
        for ticker, df in tickers_data.items():
            result = self.predict_from_ohlcv(ticker, df)
            results.append(result)
        return results


# ==============================================================================
# FASTAPI UC NOKTASI (ENDPOINT)
# ==============================================================================

try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel, Field

    # --- Request/Response Modelleri ---
    class OHLCVBar(BaseModel):
        """Tek bir mum verisi."""
        date: str = Field(..., description="Tarih (YYYY-MM-DD)")
        open: float = Field(..., description="Acilis fiyati")
        high: float = Field(..., description="En yuksek")
        low: float = Field(..., description="En dusuk")
        close: float = Field(..., description="Kapanis fiyati")
        volume: float = Field(..., description="Islem hacmi")

    class PredictRequest(BaseModel):
        """Tahmin istegi."""
        hisse: str = Field(..., description="Hisse sembolü (THYAO, GARAN...)")
        veriler: List[OHLCVBar] = Field(..., description="Son 50+ gunluk OHLCV verisi")

    class PredictResponse(BaseModel):
        """Tahmin yaniti."""
        hisse: str
        karar: str
        guven_orani: float
        tetikleyici_nedenler: List[str]
        detay: Optional[Dict[str, Any]] = None
        model_tipi: Optional[str] = None
        zaman: Optional[str] = None
        hata: Optional[str] = None

    # --- FastAPI Uygulamasi ---
    app = FastAPI(
        title="BIST AI - Canli Tahmin Motoru",
        description="ACM 465 egitilmis ML modeli ile canli borsa tahmini",
        version="1.0.0"
    )

    # Motor baslangicta bir kere yuklenir
    _engine: Optional[LiveInferenceEngine] = None

    def _get_engine() -> LiveInferenceEngine:
        global _engine
        if _engine is None:
            _engine = LiveInferenceEngine()
        return _engine

    @app.get("/", tags=["Health"])
    async def root():
        """Sunucu saglik kontrolu."""
        engine = _get_engine()
        return {
            "durum": "aktif" if engine.is_ready else "hazir_degil",
            "model_tipi": engine.model_type,
            "ozellik_sayisi": len(engine.feature_cols) if engine.feature_cols else 0,
            "mesaj": "POST /predict endpoint'ine istek gonderin."
        }

    @app.post("/predict", response_model=PredictResponse, tags=["Tahmin"])
    async def predict(request: PredictRequest):
        """
        Canli OHLCV verisinden AL/SAT tahmini yapar.

        - **hisse**: Hisse sembolü (orn: THYAO)
        - **veriler**: Son 50+ gunun OHLCV verisi (JSON array)
        """
        engine = _get_engine()

        if not engine.is_ready:
            raise HTTPException(status_code=503, detail="Model henuz yuklenmedi")

        if len(request.veriler) < 50:
            raise HTTPException(
                status_code=400,
                detail=f"En az 50 bar gerekli, {len(request.veriler)} bar gonderildi"
            )

        # JSON -> DataFrame
        records = []
        for bar in request.veriler:
            records.append({
                'Date': bar.date,
                'Open': bar.open,
                'High': bar.high,
                'Low': bar.low,
                'Close': bar.close,
                'Volume': bar.volume
            })

        df = pd.DataFrame(records)
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date').reset_index(drop=True)

        # Tahmin
        result = engine.predict_from_ohlcv(request.hisse, df)

        return PredictResponse(**result)

    @app.post("/predict/batch", tags=["Tahmin"])
    async def predict_batch(requests: List[PredictRequest]):
        """Birden fazla hisse icin toplu tahmin."""
        engine = _get_engine()
        if not engine.is_ready:
            raise HTTPException(status_code=503, detail="Model henuz yuklenmedi")

        results = []
        for req in requests:
            records = [{'Date': b.date, 'Open': b.open, 'High': b.high,
                        'Low': b.low, 'Close': b.close, 'Volume': b.volume}
                       for b in req.veriler]
            df = pd.DataFrame(records)
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.sort_values('Date').reset_index(drop=True)
            result = engine.predict_from_ohlcv(req.hisse, df)
            results.append(result)

        return results

    FASTAPI_AVAILABLE = True

except ImportError:
    FASTAPI_AVAILABLE = False
    app = None
    print("[LiveEngine] FastAPI yuklu degil. API devre disi. 'pip install fastapi uvicorn'")


# ==============================================================================
# STANDALONE CALISTIRMA
# ==============================================================================

def _demo():
    """
    Demo: Parquet cache'den bir hisse yukleyip tahmin yapar.
    Motorun calistigini dogrulamak icin kullanilir.
    """
    print("\n" + "=" * 60)
    print("  LIVE INFERENCE ENGINE - DEMO")
    print("=" * 60)

    engine = LiveInferenceEngine()

    if not engine.is_ready:
        print("\nMotor baslatılamadi. Once master_bist_ai.py'yi calistirin.")
        return

    # Demo verisi: Cache'den THYAO yukle
    cache_dir = os.path.join(BASE_DIR, 'src', 'market_data_cache')
    demo_file = os.path.join(cache_dir, 'THYAO_1d_TRY.parquet')

    if os.path.exists(demo_file):
        df = pd.read_parquet(demo_file)
        if 'symbol' in df.columns:
            df.drop(columns=['symbol'], inplace=True)

        print(f"\nDemo hisse: THYAO ({len(df)} bar)")
        result = engine.predict_from_ohlcv("THYAO", df)

        print(f"\n--- TAHMIN SONUCU ---")
        print(f"  Hisse:    {result['hisse']}")
        print(f"  Karar:    {result['karar']}")
        print(f"  Guven:    %{result['guven_orani']*100:.1f}")
        print(f"  Nedenler:")
        for r in result.get('tetikleyici_nedenler', []):
            print(f"    - {r}")
        if 'detay' in result and result['detay']:
            print(f"  Detay:")
            for k, v in result['detay'].items():
                print(f"    {k}: {v}")
    else:
        print(f"\nDemo dosyasi bulunamadi: {demo_file}")
        print("Kendi verinizle test edin: engine.predict_from_ohlcv('HISSE', df)")

    # API durumu
    if FASTAPI_AVAILABLE:
        print(f"\n  FastAPI AKTIF. Baslatmak icin:")
        print(f"    python -m uvicorn live_inference_engine:app --reload --port 8000")
        print(f"    Swagger UI: http://localhost:8000/docs")
    else:
        print(f"\n  FastAPI yuklu degil. API icin: pip install fastapi uvicorn")


if __name__ == "__main__":
    _demo()
