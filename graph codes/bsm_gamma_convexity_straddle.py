import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# --- BSM Logic ---
def bsm_price(S, K, T, r, sigma, option_type='call'):
    if T <= 0: return np.maximum(0, S - K) if option_type == 'call' else np.maximum(0, K - S)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == 'call':
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

# --- Configuration ---
S0 = 100 # Current Spot
K = 100
T = 0.5
r = 0.05
sigma = 0.2
S_range = np.linspace(80, 120, 200)

# Calculate initial prices at S0
c0 = bsm_price(S0, K, T, r, sigma, 'call')
p0 = bsm_price(S0, K, T, r, sigma, 'put')
straddle0 = c0 + p0

# 1. Long Straddle P&L (Normalized to 0 at S0)
straddle_pnl = np.array([(bsm_price(s, K, T, r, sigma, 'call') + 
                         bsm_price(s, K, T, r, sigma, 'put')) - straddle0 for s in S_range])

# 2. Pure Stock P&L (Slope 1)
stock_pnl = (S_range - S0) * 0.5 # Scale down for visualization

# 3. Combined Positions
# Gamma Scalping usually involves longing/shorting stock around the straddle
straddle_long_stock = straddle_pnl + (S_range - S0) * 0.3
straddle_short_stock = straddle_pnl - (S_range - S0) * 0.3

# --- Plotting ---
plt.style.use('default')
plt.figure(figsize=(12, 8))

# Professional Palette (Blues and Grays)
plt.plot(S_range, straddle_pnl, label='Long Straddle (Delta Neutral)', color='#084594', linewidth=3)
plt.plot(S_range, straddle_long_stock, label='Straddle + Long Stock (Bully)', color='#4292c6', linestyle='--')
plt.plot(S_range, straddle_short_stock, label='Straddle + Short Stock (Beary)', color='#9ecae1', linestyle='--')
plt.plot(S_range, S_range - S0, label='Pure Stock Position', color='gray', alpha=0.3, linestyle=':')

# Styling
plt.title(f'Gamma Scalping Analysis: Straddle & Delta Hedging\n$S_0={S0}$, K={K}, T={T}Y, $\sigma$={sigma*100}%', 
          fontsize=14, pad=20, fontweight='bold')
plt.xlabel('Stock Price (S)', fontsize=12)
plt.ylabel('Instantaneous P&L (Normalized)', fontsize=12)

# Reference lines
plt.axvline(S0, color='black', linestyle=':', alpha=0.5, label='Re-hedge Point ($S_0$)')
plt.axhline(0, color='black', linewidth=1)

# Highlights
plt.annotate('Convexity Benefit (Gamma Gain)', xy=(115, 3), xytext=(105, 5),
             arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=5),
             fontsize=10)

plt.grid(True, linestyle=':', alpha=0.4)
plt.legend(title="Strategies Components", fontsize=10, loc='best')

plt.tight_layout()

print("Graphique Gamma Scalping Straddle terminé. Affichage...")
plt.show()
