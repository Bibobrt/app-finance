import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# --- BSM Logic ---
def bsm_call_price(S, K, T, r, sigma):
    if T <= 0: return np.maximum(0, S - K)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

# --- Configuration ---
S0 = 100 # Current Spot
T = 0.5
r = 0.05
sigma = 0.2
S_range = np.linspace(70, 130, 200)

# 7 Strikes around the spot (sober selection)
strikes_val = np.linspace(80, 120, 7)
colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(strikes_val)))

# --- Plotting ---
plt.style.use('default')
plt.figure(figsize=(12, 8))

for i, K in enumerate(strikes_val):
    # Calculate initial price at S0
    p0 = bsm_call_price(S0, K, T, r, sigma)
    # P&L = P(S) - P(S0)
    pnl = np.array([bsm_call_price(s, K, T, r, sigma) - p0 for s in S_range])
    
    label = f'Strike K={K:.0f}'
    if K == 100: label += ' (ATM)'
    
    plt.plot(S_range, pnl, label=label, color=colors[i], linewidth=2)

# Styling
plt.title(f'Instantaneous Call P&L: Convexity by Strike\n$S_0={S0}$, T={T}Y, $\sigma$={sigma*100}%', 
          fontsize=14, pad=20, fontweight='bold')
plt.xlabel('Stock Price (S)', fontsize=12)
plt.ylabel(r'Instantaneous P&L ($P(S) - P(S_0)$)', fontsize=12)

# Reference lines
plt.axvline(S0, color='gray', linestyle='--', alpha=0.5, label='Current Spot')
plt.axhline(0, color='black', linewidth=1)

plt.grid(True, linestyle=':', alpha=0.4)
plt.legend(title="Strikes (Blue Gradient)", fontsize=10, loc='best', frameon=True)

plt.tight_layout()

print("Graphique P&L par Strike (Version Pro/Dégradé Bleu) terminé. Affichage...")
plt.show()
