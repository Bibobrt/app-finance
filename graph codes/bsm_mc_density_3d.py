import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import lognorm

def generate_mc_paths(S0, T, r, sigma, N_paths, N_steps):
    """
    Generate Monte Carlo paths for Geometric Brownian Motion
    under the Risk-Neutral Measure (Q).
    """
    dt = T / N_steps
    t = np.linspace(0, T, N_steps + 1)
    
    paths = np.zeros((N_steps + 1, N_paths))
    paths[0] = S0
    
    # Random normal variables (vraiment aléatoire)
    Z = np.random.standard_normal((N_steps, N_paths))
    
    mu_dt = (r - 0.5 * sigma**2) * dt
    sigma_sqrtdt = sigma * np.sqrt(dt)
    
    for i in range(1, N_steps + 1):
        paths[i] = paths[i-1] * np.exp(mu_dt + sigma_sqrtdt * Z[i-1])
        
    return t, paths

def theoretical_lognormal_pdf(S0, T, r, sigma, s_range):
    """Calculate theoretical Log-Normal PDF."""
    mu = np.log(S0) + (r - 0.5 * sigma**2) * T
    s = sigma * np.sqrt(T)
    return lognorm.pdf(s_range, s=s, scale=np.exp(mu))

# --- Configuration ---
S0 = 100.0
T = 1.0     
r = 0.05
sigma = 0.20

# 1. Paths for Visualization (small number to keep it readable)
N_paths_vis = 100
N_steps = 100

# 2. Paths for the Empirical Distribution (large number for accuracy)
# We don't save the full paths to save memory, just the final step
N_paths_dist = 50000 
_, S_final_dist = generate_mc_paths(S0, T, r, sigma, N_paths_dist, 1) # Only 1 step needed for final distribution

# Generate Visible Paths
np.random.seed(42) # For consistent visual paths, though they are mathematically random
t_mc, S_mc = generate_mc_paths(S0, T, r, sigma, N_paths_vis, N_steps)

# Generate Theoretical Q-Prob PDF
s_range = np.linspace(40, 180, 200)
pdf_q = theoretical_lognormal_pdf(S0, T, r, sigma, s_range)

# Calculation of Empirical PDF using a histogram
# We compute the histogram of the 50,000 final spots to get the empirical density
hist_vals, bin_edges = np.histogram(S_final_dist[-1], bins=100, range=(40, 180), density=True)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

# Scale PDFs for 3D visualization
pdf_max = pdf_q.max()
scaling_factor = 20.0 / pdf_max 
pdf_q_scaled = pdf_q * scaling_factor
hist_vals_scaled = hist_vals * scaling_factor

# --- 3D Plotting ---
plt.style.use('default')
fig = plt.figure(figsize=(16, 11))
ax = fig.add_subplot(111, projection='3d')

z_zeros = np.zeros(N_steps + 1)
# 1. Plot Monte Carlo Paths (Gray)
for i in range(N_paths_vis):
    if i == 0:
        ax.plot(t_mc, S_mc[:, i], z_zeros, color='gray', alpha=0.15, linewidth=0.5, label='Chemins MC simulés')
    else:
        ax.plot(t_mc, S_mc[:, i], z_zeros, color='gray', alpha=0.15, linewidth=0.5)

# 2. Identify specific paths to highlight
final_spots = S_mc[-1, :]
expected_S = S0 * np.exp(r * T)

idx_max = np.argmax(final_spots)
idx_min = np.argmin(final_spots)
idx_mean = np.argmin(np.abs(final_spots - expected_S))

# Select 2 other random paths that are not max, min or mean
available_indices = list(set(range(N_paths_vis)) - {idx_max, idx_min, idx_mean})
np.random.seed(123) # Different seed for selection
idx_randoms = np.random.choice(available_indices, 2, replace=False)

indices_to_highlight = [idx_max, idx_min, idx_mean] + list(idx_randoms)
labels = ['Pire scénario (Min)', 'Meilleur scénario (Max)', 'Proche de l\'espérance', 'Aléatoire 1', 'Aléatoire 2']

# Plot the 5 specific paths in a softer blue
for count, idx in enumerate(indices_to_highlight):
    if count == 0:
        ax.plot(t_mc, S_mc[:, idx], z_zeros, color='#1f77b4', alpha=0.7, linewidth=1.2, label='Trajectoires remarquables (Min, Max, Espérance)')
    else:
        ax.plot(t_mc, S_mc[:, idx], z_zeros, color='#1f77b4', alpha=0.7, linewidth=1.2)

# 3. Plot Theoretical PDF (Red Solid)
T_array_line = np.full_like(s_range, T)
ax.plot(T_array_line, s_range, pdf_q_scaled, color='red', linewidth=3, label="Loi log-normale théorique (Maths)")

# Fill the theoretical area
for y, z in zip(s_range, pdf_q_scaled):
    ax.plot([T, T], [y, y], [0, z], color='red', alpha=0.05)

# 4. Plot Empirical PDF from the 50,000 simulations (Green Dashed)
T_array_hist = np.full_like(bin_centers, T)
ax.plot(T_array_hist, bin_centers, hist_vals_scaled, color='green', linewidth=2.5, linestyle='--', label=f"Distribution Empirique ({N_paths_dist} MC)")

# 5. Highlight Expected Value
ax.scatter([T], [expected_S], [0], color='black', s=80, zorder=10, label=r"Espérance théorique $\mathbb{E}^Q[S_T]$")

# --- Annotations & Styling ---
# Parameter Text Box
param_text = (
    f"Paramètres du Modèle:\n"
    f"Spot Initial $(S_0)$ = {S0}\n"
    f"Maturité $(T)$ = {T} an\n"
    f"Taux sans risque $(r)$ = {r*100}%\n"
    f"Volatilité $(\sigma)$ = {sigma*100}%\n"
    f"Simulations visibles = {N_paths_vis}\n"
    f"Simul. pour l'empirique = {N_paths_dist}"
)
ax.text2D(0.02, 0.98, param_text, transform=ax.transAxes, fontsize=11, 
          verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8, edgecolor='gray'))

ax.set_title("Convergence de Monte Carlo vers la Loi Log-Normale (Q-Prob)", 
             fontsize=15, fontweight='bold', pad=20)
ax.set_xlabel('Temps (t)', fontsize=12, labelpad=15)
ax.set_ylabel('Prix du Spot (S)', fontsize=12, labelpad=15)
ax.set_zlabel('Densité de Probabilité (Scaled)', fontsize=12, labelpad=15)

ax.set_xlim(0, T * 1.05)
ax.set_ylim(40, 180)
ax.set_zlim(0, pdf_q_scaled.max() * 1.1)
ax.set_zticks([]) # Hide scaled Z ticks

ax.xaxis.set_pane_color((0.95, 0.95, 0.95, 1.0))
ax.yaxis.set_pane_color((0.95, 0.95, 0.95, 1.0))
ax.zaxis.set_pane_color((0.90, 0.90, 0.90, 1.0))

# Make grid lines more visible
ax.grid(True, color='gray', linestyle='--', linewidth=0.5, alpha=0.7)

ax.view_init(elev=25, azim=-55)
ax.legend(loc='upper right', fontsize=10, facecolor='white', framealpha=0.9)

plt.tight_layout()
print("Graphique 3D de Convergence Monte Carlo terminé. Affichage...")
plt.show()
