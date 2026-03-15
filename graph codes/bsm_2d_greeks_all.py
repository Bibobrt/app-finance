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

# Call & Put Prices
def bsm_call_price(S, K, T, r, sigma):
    d1, d2, T_safe = d1_d2(S, K, T, r, sigma)
    intrinsic = np.where(S > K, S - K, 0.0)
    price = S * norm.cdf(d1) - K * np.exp(-r * T_safe) * norm.cdf(d2)
    return np.where(T <= 1e-8, intrinsic, price)

def bsm_put_price(S, K, T, r, sigma):
    d1, d2, T_safe = d1_d2(S, K, T, r, sigma)
    intrinsic = np.where(K > S, K - S, 0.0)
    price = K * np.exp(-r * T_safe) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return np.where(T <= 1e-8, intrinsic, price)

# Call & Put Deltas
def bsm_call_delta(S, K, T, r, sigma):
    d1, _, _ = d1_d2(S, K, T, r, sigma)
    delta_expi = np.where(S > K, 1.0, np.where(S < K, 0.0, 0.5))
    return np.where(T <= 1e-8, delta_expi, norm.cdf(d1))

def bsm_put_delta(S, K, T, r, sigma):
    d1, _, _ = d1_d2(S, K, T, r, sigma)
    delta_expi = np.where(K > S, -1.0, np.where(K < S, 0.0, -0.5))
    return np.where(T <= 1e-8, delta_expi, norm.cdf(d1) - 1.0)

# Gamma (Same for Call/Put)
def bsm_gamma(S, K, T, r, sigma):
    d1, _, T_safe = d1_d2(S, K, T, r, sigma)
    nd1 = (1.0 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * d1**2)
    return np.where(T <= 1e-8, 0.0, nd1 / (S * sigma * np.sqrt(T_safe)))

# Call & Put Thetas (Normalized 1D)
def bsm_call_theta(S, K, T, r, sigma):
    d1, d2, T_safe = d1_d2(S, K, T, r, sigma)
    nd1 = (1.0 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * d1**2)
    term1 = - (S * nd1 * sigma) / (2 * np.sqrt(T_safe))
    term2 = r * K * np.exp(-r * T_safe) * norm.cdf(d2)
    return np.where(T <= 1e-8, 0.0, (term1 - term2) / 365.0)

def bsm_put_theta(S, K, T, r, sigma):
    d1, d2, T_safe = d1_d2(S, K, T, r, sigma)
    nd1 = (1.0 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * d1**2)
    term1 = - (S * nd1 * sigma) / (2 * np.sqrt(T_safe))
    term2 = r * K * np.exp(-r * T_safe) * norm.cdf(-d2)
    return np.where(T <= 1e-8, 0.0, (term1 + term2) / 365.0)

# Vega (Same for Call/Put, per 1%)
def bsm_vega(S, K, T, r, sigma):
    d1, _, T_safe = d1_d2(S, K, T, r, sigma)
    nd1 = (1.0 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * d1**2)
    return np.where(T <= 1e-8, 0.0, S * nd1 * np.sqrt(T_safe) / 100.0)

# Call & Put Rho (per 1%)
def bsm_call_rho(S, K, T, r, sigma):
    _, d2, T_safe = d1_d2(S, K, T, r, sigma)
    return np.where(T <= 1e-8, 0.0, K * T_safe * np.exp(-r * T_safe) * norm.cdf(d2) / 100.0)

def bsm_put_rho(S, K, T, r, sigma):
    _, d2, T_safe = d1_d2(S, K, T, r, sigma)
    return np.where(T <= 1e-8, 0.0, -K * T_safe * np.exp(-r * T_safe) * norm.cdf(-d2) / 100.0)


# --- Configuration ---
K = 100.0
r = 0.05
sigma = 0.20
S_range = np.linspace(60, 140, 300)

maturities_days = np.array([365, 180, 90, 30, 7, 1])
colors = cm.coolwarm(np.linspace(0.1, 0.9, len(maturities_days)))

# Functions grouped by row (Left = Call, Right = Put)
rows = [
    ("Prix", bsm_call_price, bsm_put_price),
    (r"Delta ($\Delta$)", bsm_call_delta, bsm_put_delta),
    (r"Gamma ($\Gamma$)", bsm_gamma, bsm_gamma), # Same for both
    (r"Theta 1D ($\Theta$)", bsm_call_theta, bsm_put_theta),
    (r"Vega ($\nu$)", bsm_vega, bsm_vega),       # Same for both
    (r"Rho ($\rho$)", bsm_call_rho, bsm_put_rho)
]

# --- Master 2D Plotting Matrix (6 rows x 2 cols) ---
plt.style.use('default')
fig, axes = plt.subplots(6, 2, figsize=(14, 22), sharex=True)

for row_idx, (metric_name, call_func, put_func) in enumerate(rows):
    ax_call = axes[row_idx, 0]
    ax_put  = axes[row_idx, 1]
    
    for i, days in enumerate(maturities_days):
        T = days / 365.0
        
        # Calculate data
        y_call = call_func(S_range, K, T, r, sigma)
        y_put  = put_func(S_range, K, T, r, sigma)
        
        label = f'{days}j' if days < 365 else '1 an'
        
        # Plot
        ax_call.plot(S_range, y_call, color=colors[i], label=label, linewidth=2)
        ax_put.plot(S_range, y_put, color=colors[i], label=label, linewidth=2)
        
    # Styling Call Side
    ax_call.set_title(f"{metric_name} - CALL", fontsize=12, fontweight='bold')
    ax_call.axvline(K, color='black', linestyle=':', alpha=0.5)
    ax_call.grid(True, linestyle=':', alpha=0.6)
    
    # Styling Put Side
    ax_put.set_title(f"{metric_name} - PUT", fontsize=12, fontweight='bold')
    ax_put.axvline(K, color='black', linestyle=':', alpha=0.5)
    ax_put.grid(True, linestyle=':', alpha=0.6)
    
    # Bottom labels
    if row_idx == 5:
        ax_call.set_xlabel('Spot (S)', fontsize=11)
        ax_put.set_xlabel('Spot (S)', fontsize=11)
        
    # Legend on first row only
    if row_idx == 0:
        ax_call.legend(title="Maturité", fontsize=9, loc='best')

# Super Title
fig.suptitle(r"Master Matrice 2D des Grecques (CALL vs PUT)" + f"\nStrike K={K}, Taux r={r*100}%, Volatilité $\\sigma$={sigma*100}%",
             fontsize=18, fontweight='bold', y=0.99)

plt.tight_layout(rect=[0, 0, 1, 0.98])
print("Graphique Master Matrice 2D terminé. Affichage...")
plt.show()
