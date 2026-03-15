import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.stats import norm

# --- Logic BSM Put ---
def bsm_put_price(S, K, T, r, sigma):
    """
    Calcule le prix d'un Put Black-Scholes.
    """
    # Gestion des cas limites (T=0)
    if T <= 0:
        return np.maximum(0, K - S)
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

# --- Configuration des données ---
K = 100         # Strike
r = 0.05        # Taux sans risque
sigma = 0.2     # Volatilité

# Grilles pour les axes
S_range = np.linspace(70, 130, 40)    # Spot Price (St)
T_range = np.linspace(0.01, 1.0, 40)   # Time to Maturity (T)

S_grid, T_grid = np.meshgrid(S_range, T_range)

# Calcul du Premium (Z)
Z_premium = np.vectorize(bsm_put_price)(S_grid, K, T_grid, r, sigma)

# --- Plotting Matplotlib ---
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Créer la surface (on utilise une autre colormap pour différencier du Call, ex: 'plasma')
surf = ax.plot_surface(S_grid, T_grid, Z_premium, cmap=cm.plasma,
                       linewidth=0, antialiased=True, alpha=0.9)

# Labels des axes
ax.set_xlabel('Spot Price (St)', fontsize=12, labelpad=10)
ax.set_ylabel('Time to Maturity (T)', fontsize=12, labelpad=10)
ax.set_zlabel('Put Premium (Price)', fontsize=12, labelpad=10)

# Titre
ax.set_title('Black-Scholes Put Price Surface\n(Strike K=100, r=5%, sigma=20%)', 
             fontsize=14, pad=20)

# Colorbar
fig.colorbar(surf, shrink=0.5, aspect=5, label='Option Price')

# Ajuster l'angle de vue
ax.view_init(elev=30, azim=45)

plt.tight_layout()

# Affichage direct
print("Génération du graphique Put terminé. Affichage en cours...")
plt.show()
