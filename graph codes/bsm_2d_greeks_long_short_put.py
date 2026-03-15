import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.stats import norm

# --- BSM Formulas ---
def d1_d2(S, K, T, r, sigma):
    T_safe = np.maximum(T, 1e-8)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T_safe) / (sigma * np.sqrt(T_safe))
    d2 = d1 - sigma * np.sqrt(T_safe)
    return d1, d2, T_safe

# Base Put Price
def bsm_put_price(S, K, T, r, sigma):
    d1, d2, T_safe = d1_d2(S, K, T, r, sigma)
    intrinsic = np.where(K > S, K - S, 0.0)
    price = K * np.exp(-r * T_safe) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return np.where(T <= 1e-8, intrinsic, price)

# Base Put Delta
def bsm_put_delta(S, K, T, r, sigma):
    d1, _, _ = d1_d2(S, K, T, r, sigma)
    delta_expi = np.where(K > S, -1.0, np.where(K < S, 0.0, -0.5))
    return np.where(T <= 1e-8, delta_expi, norm.cdf(d1) - 1.0)

# Base Gamma (Same for Call and Put)
def bsm_gamma(S, K, T, r, sigma):
    d1, _, T_safe = d1_d2(S, K, T, r, sigma)
    nd1 = (1.0 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * d1**2)
    return np.where(T <= 1e-8, 0.0, nd1 / (S * sigma * np.sqrt(T_safe)))

# Base Put Theta (Normalized 1D)
def bsm_put_theta(S, K, T, r, sigma):
    d1, d2, T_safe = d1_d2(S, K, T, r, sigma)
    nd1 = (1.0 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * d1**2)
    term1 = - (S * nd1 * sigma) / (2 * np.sqrt(T_safe))
    term2 = r * K * np.exp(-r * T_safe) * norm.cdf(-d2)
    return np.where(T <= 1e-8, 0.0, (term1 + term2) / 365.0)

# Base Vega (per 1%, Same for Call and Put)
def bsm_vega(S, K, T, r, sigma):
    d1, _, T_safe = d1_d2(S, K, T, r, sigma)
    nd1 = (1.0 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * d1**2)
    return np.where(T <= 1e-8, 0.0, S * nd1 * np.sqrt(T_safe) / 100.0)

# Base Put Rho (per 1%)
def bsm_put_rho(S, K, T, r, sigma):
    _, d2, T_safe = d1_d2(S, K, T, r, sigma)
    return np.where(T <= 1e-8, 0.0, -K * T_safe * np.exp(-r * T_safe) * norm.cdf(-d2) / 100.0)

# Wrappers for Short Position (Short = - Long)
def short_wrapper(func):
    return lambda S, K, T, r, sigma: -func(S, K, T, r, sigma)

# --- Configuration ---
K = 100.0
r = 0.05
sigma = 0.20
S_range = np.linspace(60, 140, 300)

maturities_days = np.array([365, 180, 90, 30, 7, 1])
colors = cm.coolwarm(np.linspace(0.1, 0.9, len(maturities_days)))

# Functions grouped by row (Left = Long Put, Right = Short Put)
rows = [
    ("Prix", bsm_put_price, short_wrapper(bsm_put_price)),
    (r"Delta ($\Delta$)", bsm_put_delta, short_wrapper(bsm_put_delta)),
    (r"Gamma ($\Gamma$)", bsm_gamma, short_wrapper(bsm_gamma)),
    (r"Theta 1D ($\Theta$)", bsm_put_theta, short_wrapper(bsm_put_theta)),
    (r"Vega ($\nu$)", bsm_vega, short_wrapper(bsm_vega)),
    (r"Rho ($\rho$)", bsm_put_rho, short_wrapper(bsm_put_rho))
]

# --- Master 2D Plotting Matrix (6 rows x 2 cols) ---
plt.style.use('default')
fig, axes = plt.subplots(6, 2, figsize=(14, 22), sharex=True)

for row_idx, (metric_name, long_func, short_func) in enumerate(rows):
    ax_long = axes[row_idx, 0]
    ax_short = axes[row_idx, 1]
    
    for i, days in enumerate(maturities_days):
        T = days / 365.0
        
        # Calculate data
        y_long = long_func(S_range, K, T, r, sigma)
        y_short = short_func(S_range, K, T, r, sigma)
        
        label = f'{days}j' if days < 365 else '1 an'
        
        # Plot
        ax_long.plot(S_range, y_long, color=colors[i], label=label, linewidth=2)
        ax_short.plot(S_range, y_short, color=colors[i], label=label, linewidth=2)
        
    # Styling Long Side
    ax_long.set_title(f"{metric_name} - LONG PUT (Acheté)", fontsize=12, fontweight='bold', color='darkgreen')
    ax_long.axvline(K, color='black', linestyle=':', alpha=0.5)
    ax_long.grid(True, linestyle=':', alpha=0.6)
    
    # Styling Short Side
    ax_short.set_title(f"{metric_name} - SHORT PUT (Vendu)", fontsize=12, fontweight='bold', color='darkred')
    ax_short.axvline(K, color='black', linestyle=':', alpha=0.5)
    ax_short.grid(True, linestyle=':', alpha=0.6)
    
    # Force Y axes to be symmetric if it's a Greek (to see the exact mirror effect)
    if row_idx > 0:
        y_max = max(abs(ax_long.get_ylim()[0]), abs(ax_long.get_ylim()[1]))
        ax_long.set_ylim(-y_max, y_max)
        ax_short.set_ylim(-y_max, y_max)
    
    # Bottom labels
    if row_idx == 5:
        ax_long.set_xlabel('Spot (S)', fontsize=11)
        ax_short.set_xlabel('Spot (S)', fontsize=11)
        
    # Legend on first row only
    if row_idx == 0:
        ax_long.legend(title="Maturité", fontsize=9, loc='upper right')

# Super Title
fig.suptitle(r"Master Matrice 2D des Grecques (LONG PUT vs SHORT PUT)" + f"\nStrike K={K}, Taux r={r*100}%, Volatilité $\\sigma$={sigma*100}%",
             fontsize=18, fontweight='bold', y=0.99)

plt.tight_layout(rect=[0, 0, 1, 0.98])
print("Graphique Matrice 2D (Long Put vs Short Put) terminé. Affichage...")
plt.show()
