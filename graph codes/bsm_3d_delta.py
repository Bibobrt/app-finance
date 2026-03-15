import numpy as np
import plotly.graph_objects as go
import os
from scipy.stats import norm

# --- BSM Delta Logic ---
def bsm_delta(S, K, T, r, sigma, option_type='call'):
    if T <= 0:
        if option_type == 'call':
            return 1.0 if S > K else 0.0
        else:
            return -1.0 if S < K else 0.0
            
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    
    if option_type == 'call':
        return norm.cdf(d1)
    else:
        return norm.cdf(d1) - 1

# --- Configuration ---
K = 100
r = 0.02
T = 1.0
S_range = np.linspace(50, 150, 50)
vol_range = np.linspace(0.05, 1.0, 50)

S_grid, vol_grid = np.meshgrid(S_range, vol_range)
Z_delta = np.array([bsm_delta(s, K, T, r, v) for s, v in zip(np.ravel(S_grid), np.ravel(vol_grid))])
Z_delta = Z_delta.reshape(S_grid.shape)

# --- Plotting ---
fig = go.Figure(data=[go.Surface(z=Z_delta, x=S_grid, y=vol_grid, colorscale='Cividis')])

fig.update_layout(
    title='Option Delta Surface (Spot vs Volatility)',
    scene=dict(
        xaxis_title='Spot Price (S)',
        yaxis_title='Volatility (sigma)',
        zaxis_title='Delta'
    ),
    template='plotly_dark',
    margin=dict(l=0, r=0, b=0, t=40)
)

# --- Save ---
output_dir = '/Users/thibaultberton/Desktop/pyhton/graphs'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

output_path = os.path.join(output_dir, 'bsm_delta_3d.html')
fig.write_html(output_path)
print(f"Graph saved to {output_path}")
