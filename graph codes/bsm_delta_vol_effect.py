import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# --- BSM Delta Logic ---
def bsm_delta(S, K, T, r, sigma):
    """
    Calculates the BSM Call Delta (N(d1)).
    """
    if T <= 0:
        return 1.0 if S > K else 0.5 if S == K else 0.0
    
    if sigma <= 0:
        # Deterministic case
        return 1.0 if S * np.exp((r)*T) > K else 0.0

    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return norm.cdf(d1)

# --- Configuration ---
K = 100
T = 0.5         # Fix maturity at 6 months
r = 0.05
S_range = np.linspace(50, 150, 200)

# Different Volatilities for comparison
vols = [
    (0.05, '5% Vol (Low)', '#1a9850'),   # Green
    (0.20, '20% Vol (Mid)', '#fee08b'),  # Yellow
    (0.50, '50% Vol (High)', '#f46d43'), # Orange
    (1.00, '100% Vol (Very High)', '#d73027') # Red
]

# --- Plotting ---
plt.style.use('default')
plt.figure(figsize=(12, 8))

for sigma, label, color in vols:
    deltas = [bsm_delta(s, K, T, r, sigma) for s in S_range]
    plt.plot(S_range, deltas, label=label, color=color, linewidth=2.5)

# Styling
plt.title(f'Effect of Volatility ($\sigma$) on Call Delta ($N(d_1)$)\nStrike K={K}, Maturity T={T}Y, r={r*100}%', 
          fontsize=14, pad=20)
plt.xlabel('Stock Price (S)', fontsize=12)
plt.ylabel('Delta ($\Delta$)', fontsize=12)

# Highlighting the Strike
plt.axvline(K, color='black', linestyle=':', alpha=0.5, label='Strike Price')
plt.axhline(0.5, color='gray', linestyle='--', alpha=0.3)

# Add Annotations
plt.annotate('Volatility flattens the Delta profile', xy=(70, 0.3), xytext=(55, 0.5),
             arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=5),
             fontsize=10)

# Grid
plt.grid(True, linestyle=':', alpha=0.6)

# Legend
plt.legend(title="Volatility levels", fontsize=10)

plt.tight_layout()

print("Graphique Delta vs Volatilité terminé. Affichage...")
plt.show()
