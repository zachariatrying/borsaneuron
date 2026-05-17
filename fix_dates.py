import pandas as pd, numpy as np, os, warnings, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
warnings.filterwarnings('ignore')

OUT = 'sunum_gorselleri'
C = {'cyan':'#00f2ff','green':'#00ff88','red':'#ff4444','yellow':'#ffbf00','bg':'#0e1117'}
plt.style.use('dark_background')

df = pd.read_csv('C:/Users/ibrah/.gemini/antigravity/scratch/bist_ai_dataset_real_30cols.csv')
feats = ['RSI_14','MACD','ATR_14','Stoch_K','Volume_Trend','Depth_Ratio','Neckline_Slope','Expert_Signal']
df_bt = df.dropna(subset=feats+['Target_T5','Max_Gain_15D','Max_Drawdown_15D']).copy()
df_bt['Date'] = pd.to_datetime(df_bt['Date'])
df_bt = df_bt.sort_values('Date')
split = int(len(df_bt)*0.8)
train_bt = df_bt.iloc[:split]

sc = StandardScaler()
xgb_bt = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42, eval_metric='logloss', verbosity=0)
xgb_bt.fit(sc.fit_transform(train_bt[feats]), train_bt['Target_T5'])

# 6 hisse
fig, axes = plt.subplots(2, 3, figsize=(20, 10), facecolor=C['bg'])
tickers = ['THYAO.IS','SISE.IS','EREGL.IS','TUPRS.IS','ASELS.IS','GARAN.IS']
for i, ticker in enumerate(tickers):
    ax = axes[i//3, i%3]
    t = df[df['Ticker']==ticker].copy()
    t['Date'] = pd.to_datetime(t['Date'])
    t = t.sort_values('Date').tail(90)
    d_min = t['Date'].iloc[0].strftime('%b %Y')
    d_max = t['Date'].iloc[-1].strftime('%b %Y')
    ax.plot(t['Date'], t['Close'], color=C['cyan'], lw=1.5)
    tf = t[feats].dropna()
    if len(tf) > 0:
        pred = xgb_bt.predict(sc.transform(tf))
        ax.scatter(t.loc[tf.index[pred==1],'Date'], t.loc[tf.index[pred==1],'Close'], c=C['green'], s=40, marker='^', zorder=5)
        ax.scatter(t.loc[tf.index[pred==0],'Date'], t.loc[tf.index[pred==0],'Close'], c=C['red'], s=25, marker='v', alpha=0.5, zorder=5)
    name = ticker.replace('.IS','')
    ax.set_title(f'{name} ({d_min} - {d_max})', fontsize=13, fontweight='bold')
    ax.set_facecolor(C['bg'])
    ax.tick_params(axis='x', rotation=30)
plt.suptitle('6 Hisse AI Sinyal Senaryosu | Veri Donemi: 2019-2024', fontsize=15, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(f'{OUT}/senaryo_6hisse.png', dpi=200, facecolor=C['bg'], bbox_inches='tight')
plt.close()
print('[OK] senaryo_6hisse.png')

# THYAO tek
t = df[df['Ticker']=='THYAO.IS'].copy()
t['Date'] = pd.to_datetime(t['Date'])
t = t.sort_values('Date').tail(90)
d_min = t['Date'].iloc[0].strftime('%d %b %Y')
d_max = t['Date'].iloc[-1].strftime('%d %b %Y')
fig, ax = plt.subplots(figsize=(14, 6), facecolor=C['bg'])
ax.plot(t['Date'], t['Close'], color=C['cyan'], lw=2, label='THYAO Kapanis')
tf = t[feats].dropna()
if len(tf) > 0:
    pred = xgb_bt.predict(sc.transform(tf))
    ax.scatter(t.loc[tf.index[pred==1],'Date'], t.loc[tf.index[pred==1],'Close'], c=C['green'], s=60, marker='^', zorder=5, label='AI: AL')
    ax.scatter(t.loc[tf.index[pred==0],'Date'], t.loc[tf.index[pred==0],'Close'], c=C['red'], s=40, marker='v', alpha=0.5, zorder=5, label='AI: BEKLE')
ax.set_title(f'THYAO AI Sinyal Senaryosu ({d_min} - {d_max})', fontsize=15, fontweight='bold')
ax.set_ylabel('Fiyat (TL)')
ax.legend(fontsize=11)
ax.set_facecolor(C['bg'])
plt.tight_layout()
plt.savefig(f'{OUT}/senaryo_thyao.png', dpi=200, facecolor=C['bg'], bbox_inches='tight')
plt.close()
print(f'THYAO: {d_min} - {d_max}')
print('[OK] senaryo_thyao.png')
