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

def bsm_call_gamma(S, K, T, r, sigma):
    if T <= 0: return 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    nd1 = (1.0 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * d1**2)
    return nd1 / (S * sigma * np.sqrt(T))

# --- Configuration ---
S0 = 100.0 # Point de couverture (Hedge Point)
K = 100.0
T = 0.5
r = 0.05
sigma = 0.20
S_range = np.linspace(70, 130, 200)

# Calculate Option Premium over the range
option_prices = np.array([bsm_call_price(s, K, T, r, sigma) for s in S_range])

# Calculate Greeks at Hedge Point
c0 = bsm_call_price(S0, K, T, r, sigma)
delta0 = bsm_call_delta(S0, K, T, r, sigma)

# Calculate Delta Hedge (Tangent line at S0)
# Equation: P_hedge(S) = P(S0) + Delta * (S - S0)
hedge_prices = c0 + delta0 * (S_range - S0)

# Calculate Greeks over the range for Subplot 2
deltas = np.array([bsm_call_delta(s, K, T, r, sigma) for s in S_range])
gammas = np.array([bsm_call_gamma(s, K, T, r, sigma) for s in S_range])

# --- Plotting ---
plt.style.use('default')
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [2, 1]}, sharex=True)

# ---> Subplot 1: Prime d'option vs Couverture Delta (L'Écart)
ax1.plot(S_range, option_prices, label='Prime d\'Option (Courbe BSM)', color='#084594', linewidth=3)
ax1.plot(S_range, hedge_prices, label=r'Couverture Delta (Tangente $\Delta$)', color='red', linestyle='--', linewidth=2)

# Shading the gap (Convexity)
ax1.fill_between(S_range, hedge_prices, option_prices, color='#4292c6', alpha=0.3, label='Écart (Convexité / Value du Gamma)')

ax1.set_title(f"Écart entre Prime d'Option et Couverture Delta\n$S_0={S0}$, K={K}, T={T}Y, $\sigma$={sigma*100}%", 
              fontsize=14, fontweight='bold', pad=15)
ax1.set_ylabel('Prix / Valeur', fontsize=12)
ax1.axvline(S0, color='black', linestyle=':', alpha=0.5, label='Point de Couverture ($S_0$)')
ax1.grid(True, linestyle=':', alpha=0.5)
ax1.legend(loc='upper left', fontsize=10)

# Annotate the gap
ax1.annotate("L'option est toujours\nau-dessus de sa tangente\n(Convexité Positive)", 
             xy=(115, bsm_call_price(115, K, T, r, sigma) - 2), 
             xytext=(105, 5),
             arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=5),
             fontsize=10, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

# ---> Subplot 2: Delta et Gamma sous-jacents
color_delta = 'darkgreen'
color_gamma = 'darkorange'

ax2.set_xlabel('Spot Price (S)', fontsize=12)
ax2.set_ylabel(r'Delta ($\Delta$)', color=color_delta, fontsize=12)
line1, = ax2.plot(S_range, deltas, color=color_delta, linewidth=2, label=r'Delta ($\Delta$)')
ax2.tick_params(axis='y', labelcolor=color_delta)
ax2.grid(True, linestyle=':', alpha=0.5)
ax2.axvline(S0, color='black', linestyle=':', alpha=0.5)

# Secondary Y-axis for Gamma
ax3 = ax2.twinx()
ax3.set_ylabel(r'Gamma ($\Gamma$)', color=color_gamma, fontsize=12)
line2, = ax3.plot(S_range, gammas, color=color_gamma, linewidth=2, linestyle='-.', label=r'Gamma ($\Gamma$)')
ax3.tick_params(axis='y', labelcolor=color_gamma)

# Legend for Subplot 2
lines = [line1, line2]
labels = [l.get_label() for l in lines]
ax2.legend(lines, labels, loc='center right', fontsize=10)

plt.tight_layout()

print("Graphique de l'écart Option vs Hedging (avec Delta/Gamma) terminé. Affichage...")
plt.show()
