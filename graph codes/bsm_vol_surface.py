import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# --- Data Simulation (Classic Quant Model) ---
strikes = np.linspace(-300, 300, 50)
expiries = np.array([0.1, 0.5, 1, 3, 5, 10, 15, 20, 30]) # Discretized for labels
# For smooth surface plotting, use a continuous range for expiries too
expiries_cont = np.linspace(0.1, 30, 50)
S_grid, E_grid = np.meshgrid(strikes, expiries_cont)

def simulate_vol(strike, expiry):
    # Base vol 20%
    base_vol = 0.20
    # Smile effect (Stronger at short maturity)
    smile_strength = 0.0000008 / (np.sqrt(expiry))
    smile = smile_strength * (strike**2)
    # Skew effect
    skew = -0.0001 * strike / (expiry**0.3)
    # Term structure decay
    term = 1.0 + 0.5 * np.exp(-0.1 * expiry)
    return (base_vol + smile + skew) * term

Z_vol = np.vectorize(simulate_vol)(S_grid, E_grid)

# --- Plotting Configuration ---
plt.style.use('default') # Classic white background
fig = plt.figure(figsize=(16, 12))
# Using GridSpec or manual positioning for insets
ax_3d = fig.add_subplot(111, projection='3d')

# 1. Main 3D Surface (Rainbow Colormap)
surf = ax_3d.plot_surface(S_grid, E_grid, Z_vol, cmap=cm.rainbow,
                          linewidth=0.1, antialiased=True, alpha=0.8)

# 2. Highlight "Cuts" on the surface (Bold Green)
# Select specific expiry and strike for cuts
cut_expiry_idx = len(expiries_cont) // 2
cut_strike_idx = len(strikes) // 2

# Smile Cut (Fix Expiry, Vary Strike)
ax_3d.plot(strikes, np.full_like(strikes, expiries_cont[cut_expiry_idx]), 
           Z_vol[cut_expiry_idx, :], color='lime', linewidth=4, zorder=10)

# Term Structure Cut (Fix Strike, Vary Expiry)
ax_3d.plot(np.full_like(expiries_cont, strikes[cut_strike_idx]), expiries_cont,
           Z_vol[:, cut_strike_idx], color='lime', linewidth=4, zorder=10)

# 3. 2D Insets (Mini-panels top corners)
# Smile Inset (Left)
ax_smile = inset_axes(ax_3d, width="25%", height="20%", loc='upper left', borderpad=3)
ax_smile.plot(strikes, Z_vol[cut_expiry_idx, :], color='lime', linewidth=2.5)
ax_smile.axvline(strikes[cut_strike_idx], color='blue', linestyle='--', alpha=0.6)
ax_smile.set_title(f"Smile ({expiries_cont[cut_expiry_idx]:.1f}Y)", fontsize=9)
ax_smile.tick_params(labelsize=7)
ax_smile.grid(alpha=0.3)

# Term Structure Inset (Right)
ax_term = inset_axes(ax_3d, width="25%", height="20%", loc='upper right', borderpad=3)
ax_term.plot(expiries_cont, Z_vol[:, cut_strike_idx], color='lime', linewidth=2.5)
ax_term.axvline(expiries_cont[cut_expiry_idx], color='blue', linestyle='--', alpha=0.6)
ax_term.set_title(f"Term Structure (ATM)", fontsize=9)
ax_term.tick_params(labelsize=7)
ax_term.grid(alpha=0.3)

# 4. Styling the 3D Box
ax_3d.set_title('QUANTITATIVE VOLATILITY CUBE', fontsize=18, pad=40, fontweight='bold')

# Labels and Ticks
ax_3d.set_xlabel('Strike (bps)', fontsize=11, labelpad=15)
ax_3d.set_xticks([-300, -150, 0, 150, 300])
ax_3d.set_xticklabels(['-300', '-150', 'ATM', '150', '300'])

ax_3d.set_ylabel('Expiry (Years)', fontsize=11, labelpad=15)
ax_3d.set_yticks([0.1, 5, 10, 20, 30])
ax_3d.set_yticklabels(['1M', '5Y', '10Y', '20Y', '30Y'])

ax_3d.set_zlabel('Implied Volatility', fontsize=11, labelpad=15)

# Panes and Grid
ax_3d.xaxis.set_pane_color((0.95, 0.95, 0.95, 1.0))
ax_3d.yaxis.set_pane_color((0.95, 0.95, 0.95, 1.0))
ax_3d.zaxis.set_pane_color((0.9, 0.9, 0.9, 1.0))
ax_3d.grid(True, linestyle=':', alpha=0.5)

# View Angle
ax_3d.view_init(elev=25, azim=-125)

# Colorbar
cbar = fig.colorbar(surf, ax=ax_3d, shrink=0.5, aspect=15, pad=0.1)
cbar.set_label('Volatility Level', fontsize=10)

plt.tight_layout()

print("High-Fidelity Quantitative Volatility Cube complete. Displaying...")
plt.show()
