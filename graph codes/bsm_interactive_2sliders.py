import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from scipy.stats import norm

# --- BSM Formulas ---
def d1_d2(S, K, T, r, sigma):
    T_safe = np.maximum(T, 1e-8)
    sigma_safe = np.maximum(sigma, 1e-8)
    d1 = (np.log(S / K) + (r + 0.5 * sigma_safe**2) * T_safe) / (sigma_safe * np.sqrt(T_safe))
    d2 = d1 - sigma_safe * np.sqrt(T_safe)
    return d1, d2, T_safe, sigma_safe

def bsm_call_price(S, K, T, r, sigma):
    d1, d2, T_safe, _ = d1_d2(S, K, T, r, sigma)
    intrinsic = np.where(S > K, S - K, 0.0)
    price = S * norm.cdf(d1) - K * np.exp(-r * T_safe) * norm.cdf(d2)
    return np.where(T <= 1e-8, intrinsic, price)

def bsm_call_delta(S, K, T, r, sigma):
    d1, _, _, _ = d1_d2(S, K, T, r, sigma)
    delta_expi = np.where(S > K, 1.0, np.where(S < K, 0.0, 0.5))
    return np.where(T <= 1e-8, delta_expi, norm.cdf(d1))

def bsm_gamma(S, K, T, r, sigma):
    d1, _, T_safe, sigma_safe = d1_d2(S, K, T, r, sigma)
    nd1 = (1.0 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * d1**2)
    return np.where(T <= 1e-8, 0.0, nd1 / (S * sigma_safe * np.sqrt(T_safe)))

def bsm_call_theta(S, K, T, r, sigma):
    d1, d2, T_safe, sigma_safe = d1_d2(S, K, T, r, sigma)
    nd1 = (1.0 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * d1**2)
    term1 = - (S * nd1 * sigma_safe) / (2 * np.sqrt(T_safe))
    term2 = r * K * np.exp(-r * T_safe) * norm.cdf(d2)
    return np.where(T <= 1e-8, 0.0, (term1 - term2) / 365.0)

def bsm_vega(S, K, T, r, sigma):
    d1, _, T_safe, _ = d1_d2(S, K, T, r, sigma)
    nd1 = (1.0 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * d1**2)
    return np.where(T <= 1e-8, 0.0, S * nd1 * np.sqrt(T_safe) / 100.0)

def bsm_call_rho(S, K, T, r, sigma):
    _, d2, T_safe, _ = d1_d2(S, K, T, r, sigma)
    return np.where(T <= 1e-8, 0.0, K * T_safe * np.exp(-r * T_safe) * norm.cdf(d2) / 100.0)


# --- Configuration ---
K = 100.0
r = 0.05
init_sigma = 0.20 # Initial Volatility: 20%
init_T_days = 90.0 # Initial Maturity: 90 days
S_range = np.linspace(60, 140, 400) # Increased resolution for sharp peaks

# --- Setup Interactive Plot ---
plt.style.use('default')
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
# Leave space at the bottom for TWO sliders
plt.subplots_adjust(bottom=0.25, hspace=0.3)
axes = axes.flatten()

titles_and_funcs = [
    ("Prix (Call)", bsm_call_price),
    (r"Delta ($\Delta$)", bsm_call_delta),
    (r"Gamma ($\Gamma$)", bsm_gamma),
    (r"Theta 1D ($\Theta$)", bsm_call_theta),
    (r"Vega ($\nu$)", bsm_vega),
    (r"Rho ($\rho$)", bsm_call_rho)
]

# Dictionary to store the single line object for each subplot
line_objects = {}

# Initial Drawing
for idx, (title, func) in enumerate(titles_and_funcs):
    ax = axes[idx]
    
    # Calculate initial data
    Z_data = func(S_range, K, init_T_days / 365.0, r, init_sigma)
    
    # Plot ONE single dynamic line
    line, = ax.plot(S_range, Z_data, color='#1f77b4', linewidth=2.5, label='Profil actuel')
    line_objects[idx] = line
    
    ax.set_title(title, fontsize=12, fontweight='bold', color='darkgreen')
    ax.set_xlabel('Spot (S)', fontsize=10)
    ax.axvline(K, color='black', linestyle=':', alpha=0.5, label='Strike')
    ax.grid(True, linestyle=':', alpha=0.5)
    
    # Specific zero lines
    if "Theta" in title or "Rho" in title:
        ax.axhline(0, color='gray', linestyle='--', linewidth=1)
        
    if idx == 0:
        ax.legend(fontsize=9, loc='upper left')

# Super Title
suptitle = fig.suptitle(r"Dashboard Interactif: Temps & Volatilité (LONG CALL)" + f"\nStrike K={K}, Taux r={r*100}%",
             fontsize=16, fontweight='bold', y=0.97, color='darkgreen')

# --- Add 2 Sliders ---
# [left, bottom, width, height]
ax_vol = plt.axes([0.25, 0.12, 0.50, 0.03])  
ax_time = plt.axes([0.25, 0.05, 0.50, 0.03])

vol_slider = Slider(
    ax=ax_vol,
    label='Volatilité Implicite ($\sigma$)',
    valmin=0.01,   # 1%
    valmax=1.50,   # 150%
    valinit=init_sigma,
    valfmt='%1.0f%%', 
    color='darkorange'
)

time_slider = Slider(
    ax=ax_time,
    label='Maturité (Jours)',
    valmin=0.0,    # 0 days (expiration)
    valmax=365.0,  # 1 year
    valinit=init_T_days,
    valfmt='%1.0f j', 
    color='dodgerblue'
)

# Custom formatting functions
def update_sliders_text(sigma_val, time_val):
    vol_slider.valtext.set_text(f'{sigma_val*100:.0f}%')
    time_slider.valtext.set_text(f'{time_val:.0f} Jours')

# Update function called when ANY slider moves
def update(val=None):
    current_sigma = vol_slider.val
    current_time_days = time_slider.val
    
    # Format texts cleanly
    update_sliders_text(current_sigma, current_time_days)
    
    T = current_time_days / 365.0
    
    for idx, (title, func) in enumerate(titles_and_funcs):
        ax = axes[idx]
        
        # Calculate new data
        new_Z_data = func(S_range, K, T, r, current_sigma)
        
        # Update the single line
        line_objects[idx].set_ydata(new_Z_data)
        
        # Danger zone: Recompute axis limits so sharp peaks (like Gamma at T=0) don't go off-screen
        ax.relim()
        ax.autoscale_view()
        
    fig.canvas.draw_idle()

# Connect both sliders to the SAME update function
vol_slider.on_changed(update)
time_slider.on_changed(update)

# Initial text format
update_sliders_text(init_sigma, init_T_days)

print("Double Dashboard interactif prêt. Déplacez les curseurs en bas (Volatilité et Temps).")
plt.show()
