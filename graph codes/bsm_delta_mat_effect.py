import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# --- BSM Delta Logic ---
def bsm_delta(S, K, T, r, sigma):
    if T <= 1e-6: # At expiration
        return 1.0 if S > K else 0.5 if S == K else 0.0
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return norm.cdf(d1)

# --- Configuration ---
K = 100
r = 0.02
sigma = 0.2
S_range = np.linspace(80, 120, 200)

# Different times to maturity for color coding
# From 1 Year to 1 Day
tenors = [
    (1.0, '1 Year', '#08306b'),    # Dark Blue
    (0.5, '6 Months', '#2171b5'),  # Blue
    (0.1, '1 Month', '#6baed6'),   # Light Blue
    (0.01, '1 Week', '#fee0d2'),   # Very Light (Transition)
    (0.002, '1 Day', '#de2d26'),    # Red (Aggressive)
    (0.0, 'Expiration (T=0)', 'black') # Black Step
]

# --- Plotting ---
plt.style.use('default')
plt.figure(figsize=(12, 8))

for T, label, color in tenors:
    deltas = [bsm_delta(s, K, T, r, sigma) for s in S_range]
    lw = 2.5 if T <= 0.002 else 1.5
    ls = '--' if T == 0 else '-'
    plt.plot(S_range, deltas, label=label, color=color, linewidth=lw, linestyle=ls)

# Styling
plt.title(f'Effect of Time to Maturity on Call Delta\nStrike K={K}, sigma={sigma*100}%', 
          fontsize=14, pad=20)
plt.xlabel('Stock Price (S)', fontsize=12)
plt.ylabel('Delta ($\Delta$)', fontsize=12)

# Highlighting the Strike (ATM Line)
plt.axvline(K, color='gray', linestyle=':', alpha=0.5)
plt.axhline(0.5, color='gray', linestyle=':', alpha=0.5)

# Add Labels
plt.text(82, 0.9, 'Deep ITM (Delta $\\to$ 1)', fontsize=10, color='gray')
plt.text(82, 0.1, 'Deep OTM (Delta $\\to$ 0)', fontsize=10, color='gray')

# Grid
plt.grid(True, linestyle=':', alpha=0.4)

# Legend
plt.legend(title="Time to Maturity", fontsize=10, loc='lower right')

plt.tight_layout()

print("Graphique Delta vs Maturité terminé. Affichage...")
plt.show()
