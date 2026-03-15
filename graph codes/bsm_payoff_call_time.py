import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# --- BSM Logic ---
def bsm_call_price(S, K, T, r, sigma):
    if T <= 1e-6: # At expiration
        return np.maximum(0, S - K)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

# --- Configuration ---
K = 100
r = 0.05
sigma = 0.2
S_range = np.linspace(60, 140, 200)

# Different times to maturity
tenors = [
    (1.0, '1 Year', '#1f77b4'),   # Blue
    (0.5, '6 Months', '#ff7f0e'), # Orange
    (0.1, '1 Month', '#2ca02c'),  # Green
    (0.0, 'Payoff (T=0)', 'red')  # Red Bold
]

# --- Plotting ---
plt.style.use('default')
plt.figure(figsize=(12, 8))

for T, label, color in tenors:
    prices = [bsm_call_price(s, K, T, r, sigma) for s in S_range]
    linewidth = 3 if T == 0 else 2
    linestyle = '--' if T == 0 else '-'
    plt.plot(S_range, prices, label=label, color=color, linewidth=linewidth, linestyle=linestyle)

# Styling
plt.title(f'Call Option Value vs Time to Maturity (Time Value Decay)\nStrike K={K}, r={r*100}%, sigma={sigma*100}%', 
          fontsize=14, pad=20)
plt.xlabel('Stock Price (S)', fontsize=12)
plt.ylabel('Option Price', fontsize=12)

# Highlighting the Strike
plt.axvline(K, color='black', linestyle=':', alpha=0.5, label='Strike Price')

# Add Grid
plt.grid(True, linestyle=':', alpha=0.6)

# Legend
plt.legend(fontsize=10)

# Annotations for Time Value
plt.annotate('Intrinsic Value', xy=(120, 20), xytext=(125, 10),
             arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=5),
             fontsize=10)

plt.tight_layout()

print("Graphique Payoff Call vs Time Value terminé. Affichage...")
plt.show()
