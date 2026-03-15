import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# --- BSM Logic ---
def bsm_price(S, K, T, r, sigma, option_type='call'):
    if T <= 1e-6: # At expiration
        return np.maximum(0, S - K) if option_type == 'call' else np.maximum(0, K - S)
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type == 'call':
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

# --- Configuration ---
K = 100
r = 0.05
sigma = 0.2
S_range = np.linspace(60, 140, 200)

# Time to maturity colors/labels
tenors = [
    (1.0, '1 Year', '#1f77b4'),
    (0.2, ' ~3 Months', '#ff7f0e'),
    (0.0, 'Payoff (T=0)', 'red')
]

# --- Plotting 2x2 Grid ---
plt.style.use('default')
fig, axes = plt.subplots(2, 2, figsize=(16, 12), sharex=True)

# Positions: (Row, Col, Type, Side)
configs = [
    (0, 0, 'call', 1,  'Long Call (Buy)'),
    (0, 1, 'put',  1,  'Long Put (Buy)'),
    (1, 0, 'call', -1, 'Short Call (Sell)'),
    (1, 1, 'put',  -1, 'Short Put (Sell)')
]

for row, col, otype, side, title in configs:
    ax = axes[row, col]
    for T, label, color in tenors:
        prices = np.array([bsm_price(s, K, T, r, sigma, otype) for s in S_range])
        values = side * prices
        
        lw = 2.5 if T == 0 else 1.5
        ls = '--' if T == 0 else '-'
        ax.plot(S_range, values, label=label, color=color, linewidth=lw, linestyle=ls)
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.axvline(K, color='black', linestyle=':', alpha=0.4)
    ax.axhline(0, color='black', linewidth=1)
    ax.grid(True, linestyle=':', alpha=0.6)
    if col == 0: ax.set_ylabel('Profit / Loss', fontsize=12)
    if row == 1: ax.set_xlabel('Stock Price (S)', fontsize=12)
    ax.legend(fontsize=9)

plt.suptitle(f'Option Payoff & Time Value Analysis (4 Quadrants)\nStrike K={K}, r={r*100}%, sigma={sigma*100}%', 
             fontsize=18, y=0.95)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])

print("Génération du quad-graphique (Long/Short Call/Put) terminée. Affichage...")
plt.show()
