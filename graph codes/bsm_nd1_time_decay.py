import numpy as np
import matplotlib.pyplot as plt

# --- BSM n'(d1) Logic ---
def bsm_nd1(S, K, T, r, sigma):
    """
    Calculates the BSM n'(d1) probability density.
    This is proportional to Vega and Gamma.
    """
    if T <= 0:
        return 0.0
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    nd1 = (1.0 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * d1**2)
    return nd1

# --- Configuration ---
K = 100
r = 0.05
sigma = 0.2
T_range = np.linspace(0.001, 2.0, 300) # From nearly 0 to 2 years

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
    densities = [bsm_nd1(S, K, t, r, sigma) for t in T_range]
    plt.plot(T_range, densities, label=label, color=color, linewidth=2.5)

# Styling
plt.title(f"Evolution of $n'(d_1)$ over Maturity (Time Decay)\nStrike K={K}, r={r*100}%, sigma={sigma*100}%", 
          fontsize=14, pad=20)
plt.xlabel('Time to Maturity (Years)', fontsize=12)
plt.ylabel(r"Probability Density $n'(d_1)$", fontsize=12)

# Grid
plt.grid(True, linestyle=':', alpha=0.5)

# Ticks
plt.xticks(np.arange(0, 2.1, 0.25))

# Legend
plt.legend(title="Spot Scenarios", fontsize=10, loc='best')

# Annotations
plt.annotate(r"ATM density explosion as T $\to$ 0", xy=(0.05, 0.4), xytext=(0.4, 0.35),
             arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=5),
             fontsize=10)

plt.tight_layout()

print("Graphique n'(d1) vs Maturité (2D) terminé. Affichage...")
plt.show()
