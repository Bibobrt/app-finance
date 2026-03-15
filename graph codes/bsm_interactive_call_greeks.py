import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
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
init_sigma = 0.20 # Initial Volatility
S_range = np.linspace(60, 140, 200)

maturities_days = np.array([365, 180, 90, 30, 7, 1])
colors = cm.coolwarm(np.linspace(0.1, 0.9, len(maturities_days)))

# --- Setup Interactive Plot ---
plt.style.use('default')
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
# Leave space at the bottom for the slider
plt.subplots_adjust(bottom=0.15, hspace=0.3)
axes = axes.flatten()

titles_and_funcs = [
    ("Prix (Call)", bsm_call_price),
    (r"Delta ($\Delta$)", bsm_call_delta),
    (r"Gamma ($\Gamma$)", bsm_gamma),
    (r"Theta 1D ($\Theta$)", bsm_call_theta),
    (r"Vega ($\nu$)", bsm_vega),
    (r"Rho ($\rho$)", bsm_call_rho)
]

# Dictionary to store the line objects so we can update them later
line_objects = {idx: [] for idx in range(len(titles_and_funcs))}

# Initial Drawing
for idx, (title, func) in enumerate(titles_and_funcs):
    ax = axes[idx]
    
    for i, days in enumerate(maturities_days):
        T = days / 365.0
        Z_data = func(S_range, K, T, r, init_sigma)
        
        label = f'{days}j' if days < 365 else '1 an'
        # Plot and save the Line2D object
        line, = ax.plot(S_range, Z_data, color=colors[i], label=label, linewidth=2)
        line_objects[idx].append(line)
        
    ax.set_title(title, fontsize=12, fontweight='bold', color='darkgreen')
    ax.set_xlabel('Spot (S)', fontsize=10)
    ax.axvline(K, color='black', linestyle=':', alpha=0.5)
    ax.grid(True, linestyle=':', alpha=0.5)
    
    # Specific zero lines
    if "Theta" in title or "Rho" in title:
        ax.axhline(0, color='gray', linestyle='--', linewidth=1)
        
    if idx == 0:
        ax.legend(title="Maturité", fontsize=9, loc='upper left')

# Super Title
suptitle = fig.suptitle(r"Dashboard Interactif des Grecques (LONG CALL)" + f"\nStrike K={K}, Taux r={r*100}%",
             fontsize=16, fontweight='bold', y=0.97, color='darkgreen')

# --- Add Slider ---
ax_slider = plt.axes([0.25, 0.05, 0.50, 0.03]) # [left, bottom, width, height]
vol_slider = Slider(
    ax=ax_slider,
    label='Volatilité Implicite ($\sigma$)',
    valmin=0.01,   # 1%
    valmax=1.50,   # 150%
    valinit=init_sigma,
    valfmt='%1.0f%%', 
    color='darkorange'
)

# Custom formatting function for the slider text so it shows exact percentages
def update_slider_text(val):
    vol_slider.valtext.set_text(f'{val*100:.0f}%')

# Update function called when slider moves
def update(val):
    current_sigma = vol_slider.val
    update_slider_text(current_sigma)
    
    # Calculate global min/max for dynamic Y-axis scaling if needed (optional, here we keep auto-scaling but recompute it)
    for idx, (title, func) in enumerate(titles_and_funcs):
        ax = axes[idx]
        
        for i, days in enumerate(maturities_days):
            T = days / 365.0
            # Calculate new data
            new_Z_data = func(S_range, K, T, r, current_sigma)
            # Update the line object
            line_objects[idx][i].set_ydata(new_Z_data)
            
        # Recompute axis limits so the curves don't go off-screen
        ax.relim()
        ax.autoscale_view()
        
    fig.canvas.draw_idle()

# Connect the slider to the update function
vol_slider.on_changed(update)

# Initial text format to percentage
update_slider_text(init_sigma)

print("Tableau de bord interactif prêt. Déplacez le curseur en bas pour changer la volatilité.")
plt.show()
