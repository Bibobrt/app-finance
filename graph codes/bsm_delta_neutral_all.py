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

def bsm_delta(S, K, T, r, sigma, option_type='call'):
    if T <= 0: return 1.0 if S > K else 0.0 if option_type == 'call' else -1.0 if S < K else 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    if option_type == 'call':
        return norm.cdf(d1)
    else:
        return norm.cdf(d1) - 1.0

# --- Configuration ---
S0 = 100 # Initial Spot (hedge point)
K = 100
T = 0.5
r = 0.02
sigma = 0.2
S_range = np.linspace(80, 120, 200)

# --- Plotting 2x2 Grid ---
plt.style.use('default')
fig, axes = plt.subplots(2, 2, figsize=(16, 12), sharex=True)

# Positions: (Row, Col, Type, Side, Title)
configs = [
    (0, 0, 'call', 1,  'Long Call (Hedged)'),
    (0, 1, 'put',  1,  'Long Put (Hedged)'),
    (1, 0, 'call', -1, 'Short Call (Hedged)'),
    (1, 1, 'put',  -1, 'Short Put (Hedged)')
]

for row, col, otype, side, title in configs:
    ax = axes[row, col]
    
    # Calculate price and delta at hedge point S0
    p0 = bsm_price(S0, K, T, r, sigma, otype)
    d0 = bsm_delta(S0, K, T, r, sigma, otype)
    
    # Portfolio Price across S range (Normalized to 0 at S0)
    option_pnl = side * (np.array([bsm_price(s, K, T, r, sigma, otype) for s in S_range]) - p0)
    
    # Hedging Component: side * -Delta * (S - S0)
    # If Long Call (side=1, d0=0.5): we sell 0.5 shares -> -0.5 * (S - S0)
    # If Short Call (side=-1, d0=0.5): option delta is -0.5, so we buy +0.5 shares -> +0.5 * (S-S0)
    hedge_pnl = side * (-d0 * (S_range - S0))
    
    # Total Portfolio P&L
    total_pnl = option_pnl + hedge_pnl
    
    # Plotting
    ax.plot(S_range, total_pnl, label='Hedged Portfolio', color='purple', linewidth=3)
    ax.plot(S_range, option_pnl, label='Option P&L', color='blue', linestyle='--', alpha=0.4)
    ax.plot(S_range, hedge_pnl, label='Hedge P&L', color='red', linestyle='--', alpha=0.4)
    
    # Styling
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.axvline(S0, color='gray', linestyle=':', alpha=0.5)
    ax.axhline(0, color='black', linewidth=1)
    ax.grid(True, linestyle=':', alpha=0.5)
    
    if col == 0: ax.set_ylabel('Portfolio P&L', fontsize=12)
    if row == 1: ax.set_xlabel('Stock Price (S)', fontsize=12)
    ax.legend(fontsize=9)

plt.suptitle(f'Delta-Neutral Portfolios (4 Quadrants)\nHedge at $S_0={S0}$, Strike K={K}, T={T}Y, $\sigma$={sigma*100}%', 
             fontsize=18, y=0.95)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])

print("Génération du quad-graphique Delta-Neutre terminée. Affichage...")
plt.show()
