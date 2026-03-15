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
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return norm.cdf(d1)

# --- Configuration ---
K = 100
r = 0.05
sigma = 0.2
T_range = np.linspace(0.001, 2.0, 200) # From ~0 to 2 years

# Different Spot scenarios
scenarios = [
    (110, 'In-the-Money (S=110)', '#1a9850'),    # Green
    (105, 'Slightly ITM (S=105)', '#a6d96a'),   # Light Green
    (100, 'At-the-Money (S=100)', '#fee08b'),   # Yellow
    (95,  'Slightly OTM (S=95)',  '#fdae61'),   # Orange
    (90,  'Out-of-the-Money (S=90)', '#d73027') # Red
]

# --- Plotting ---
plt.style.use('default')
plt.figure(figsize=(12, 8))

for S, label, color in scenarios:
    deltas = [bsm_delta(S, K, t, r, sigma) for t in T_range]
    plt.plot(T_range, deltas, label=label, color=color, linewidth=2.5)

# Styling
plt.title(f'Evolution of Call Delta over Maturity (Time Decay)\nStrike K={K}, r={r*100}%, sigma={sigma*100}%', 
          fontsize=14, pad=20)
plt.xlabel('Time to Maturity (Years)', fontsize=12)
plt.ylabel(r'Delta ($\Delta$)', fontsize=12) # Raw string to avoid warning

# Ticks
plt.xticks(np.arange(0, 2.1, 0.25))
plt.yticks(np.arange(0, 1.1, 0.1))

# Higlighting the 0.5 Line
plt.axhline(0.5, color='gray', linestyle='--', alpha=0.3)

# Grid
plt.grid(True, linestyle=':', alpha=0.5)

# Legend
plt.legend(title="Spot Scenarios", fontsize=10, loc='best')

# Annotations
plt.annotate('Convergence towards 1 or 0 as T $\\to$ 0', xy=(0.05, 0.5), xytext=(0.4, 0.2),
             arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=5),
             fontsize=10)

plt.tight_layout()

print("Graphique Delta vs Maturité (Maturité en abscisse) terminé. Affichage...")
plt.show()
