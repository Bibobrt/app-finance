import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# --- BSM Logic ---
def bsm_call_price(S, K, T, r, sigma):
    if T <= 0: return np.maximum(0, S - K)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

def bsm_call_delta(S, K, T, r, sigma):
    if T <= 0: return 1.0 if S > K else 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return norm.cdf(d1)

# --- Configuration ---
S0 = 100 # Initial Spot (at which we hedge)
K = 100
T = 0.5
r = 0.02
sigma = 0.2
S_range = np.linspace(80, 120, 200)

# 1. Option Price Profile
option_prices = np.array([bsm_call_price(s, K, T, r, sigma) for s in S_range])

# 2. Delta Hedge Component
# We sell delta shares at price S0
delta0 = bsm_call_delta(S0, K, T, r, sigma)
# The hedge value change is -Delta * (S - S0)
hedge_values = -delta0 * (S_range - S0)

# 3. Delta-Neutral Portfolio Profile
# Portfolio Value = Option Price + Cash/Hedge
# We normalize it so the value is 0 at the hedge point S0
portfolio_values = (option_prices - bsm_call_price(S0, K, T, r, sigma)) + hedge_values

# --- Plotting ---
plt.style.use('default')
plt.figure(figsize=(12, 8))

plt.plot(S_range, portfolio_values, label='Delta-Neutral Portfolio P&L', color='purple', linewidth=3)
plt.plot(S_range, (option_prices - bsm_call_price(S0, K, T, r, sigma)), label='Long Call P&L (Unhedged)', color='blue', linestyle='--', alpha=0.5)
plt.plot(S_range, hedge_values, label=f'Short Hedge ($-{delta0:.2f} \cdot \Delta S$)', color='red', linestyle='--', alpha=0.5)

# Styling
plt.title(r'Delta-Neutral Call Profile (Delta Hedging at $S_0=100$)' + f'\nStrike K={K}, T={T}Y, $\sigma$={sigma*100}%', 
          fontsize=14, pad=20)
plt.xlabel('Stock Price (S)', fontsize=12)
plt.ylabel('Portfolio P&L', fontsize=12)

# Reference lines
plt.axvline(S0, color='black', linestyle=':', alpha=0.6, label='Hedge Point ($S_0$)')
plt.axhline(0, color='black', linewidth=1)

# Annotations for Convexity
plt.annotate('Pure Convexity (Gamma Gain)', xy=(115, 2.5), xytext=(105, 4),
             arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=5),
             fontsize=11, fontweight='bold')

plt.grid(True, linestyle=':', alpha=0.5)
plt.legend(fontsize=10)

plt.tight_layout()

print("Graphique Profile Delta-Neutre terminé. Affichage...")
plt.show()
