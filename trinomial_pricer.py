import numpy as np

class TrinomialOptionPricer:
    def __init__(self, S, K, T, r, sigma, N):
        """
        Initializes the Trinomial Option Pricer using Kamrad-Ritchken calibration.

        Parameters:
        S (float): Spot price
        K (float): Strike price
        T (float): Time to maturity (in years)
        r (float): Risk-free interest rate (annual)
        sigma (float): Volatility (annual)
        N (int): Number of time steps
        """
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.N = int(N)

        # Calculated parameters
        self.dt = T / self.N
        
        # Kamrad-Ritchken parameters (lambda = sqrt(3))
        # u = exp(sigma * sqrt(3 * dt))
        # d = 1/u
        self.u = np.exp(sigma * np.sqrt(3 * self.dt))
        self.d = 1 / self.u
        self.m = 1.0 # Middle jump (no change in log-space relative to drift, strictly speaking m=1 in this model formulation for price multiplier)

        # Probabilities
        # v = r - 0.5 * sigma^2
        # pu = 1/6 + (v * sqrt(dt)) / (2 * sigma * sqrt(3))
        # pm = 2/3
        # pd = 1/6 - (v * sqrt(dt)) / (2 * sigma * sqrt(3))
        
        # Note: Some formulations use r in v, others use r-q. Assuming q=0 (no dividends).
        # Also, checking stability condition: probabilities must be between 0 and 1.
        
        nu = r - 0.5 * sigma**2
        sqrt_dt = np.sqrt(self.dt)
        
        self.pu = (1/6) + (nu * sqrt_dt) / (2 * sigma * np.sqrt(3))
        self.pm = (2/3)
        self.pd = (1/6) - (nu * sqrt_dt) / (2 * sigma * np.sqrt(3))
        
        # Stability check: Probabilities must be within [0, 1]
        if not (0 <= self.pu <= 1 and 0 <= self.pm <= 1 and 0 <= self.pd <= 1):
            raise ValueError(
                "Calculated probabilities are outside [0, 1], leading to numerical instability.\n"
                f"pu={self.pu:.6f}, pm={self.pm:.6f}, pd={self.pd:.6f}\n"
                "Condition failed. Try increasing N (number of steps) or adjusting parameters."
            )
        
        self.discount = np.exp(-r * self.dt)

    def _initialize_tree(self):
        """
        Initializes the asset prices at maturity (Step N).
        In a trinomial tree, at step N, there are 2*N + 1 nodes.
        The price at node i (where i goes from 0 to 2N) corresponds to S * u^(N-i).
        Actually, let's correspond indices more carefully to number of 'up' moves vs 'down' moves.
        
        Let j be the node index at time N, ranging from 0 to 2N.
        j=0: Max Up moves (N ups). Price = S * u^N
        j=N: Middle (0 net moves). Price = S
        j=2N: Max Down moves (N downs). Price = S * u^(-N) = S * d^N
        
        General formula at step n (where n is 0 to N):
        Nodes range from 0 to 2n.
        Price at node i (0 <= i <= 2n): S * u^(n - i)
        """
        num_nodes = 2 * self.N + 1
        asset_prices = np.zeros(num_nodes)
        
        # Initialize prices at maturity (step N)
        for i in range(num_nodes):
            # Exponent for u is (N - i)
            # if i=0 (top): u^N
            # if i=N (mid): u^0 = 1
            # if i=2N (bot): u^-N = d^N
            asset_prices[i] = self.S * (self.u ** (self.N - i))
            
        return asset_prices

    def price(self, option_type='call', style='european'):
        """
        Calculates the option price using the Trinomial tree model.
        
        Parameters:
        option_type (str): 'call' or 'put'
        style (str): 'european' or 'american'
        
        Returns:
        float: Calculated option price
        """
        option_type = option_type.lower()
        style = style.lower()
        
        # Initialize asset prices at maturity
        asset_prices = self._initialize_tree()
        
        # Initialize option values at maturity
        num_nodes_maturity = 2 * self.N + 1
        option_values = np.zeros(num_nodes_maturity)
        
        for i in range(num_nodes_maturity):
            if option_type == 'call':
                option_values[i] = max(0, asset_prices[i] - self.K)
            else: # put
                option_values[i] = max(0, self.K - asset_prices[i])
        
        # Backward induction
        # j represents the time step, going from N-1 down to 0
        for j in range(self.N - 1, -1, -1):
            # At step j, there are 2*j + 1 nodes.
            # We compute values for indices 0 to 2*j.
            # For a node at index i in step j, the children in step j+1 are at:
            # i (up), i+1 (mid), i+2 (down)
            
            # Since we can do this in vectorised way or loop. 
            # We are writing in a loop for clarity, but slicing would be faster.
            # Let's stick to the loop size for safety, but we can reuse the option_values array
            # or create a temporary one. A temp one is safer to avoid overwriting needed values during the loop if done in-place purely.
            # However, since i depends on i, i+1, i+2 of the *previous* array (next time step), 
            # and we are essentially shrinking the array, we can just write to the first 2*j+1 elements?
            # Actually, `option_values` currently holds values for step j+1.
            # We want to calculate values for step j.
            # value[i] = df * (pu * old[i] + pm * old[i+1] + pd * old[i+2])
            
            current_layer_size = 2 * j + 1
            new_values = np.zeros(current_layer_size)
            
            for i in range(current_layer_size):
                # Continuation value
                continuation = self.discount * (
                    self.pu * option_values[i] + 
                    self.pm * option_values[i+1] + 
                    self.pd * option_values[i+2]
                )
                
                if style == 'american':
                    # Calculate underlying price at this node (j, i)
                    # S_node = S * u^(j - i)
                    S_node = self.S * (self.u ** (j - i))
                    
                    if option_type == 'call':
                        intrinsic = max(0, S_node - self.K)
                    else:
                        intrinsic = max(0, self.K - S_node)
                        
                    new_values[i] = max(intrinsic, continuation)
                else:
                    new_values[i] = continuation
            
            # Update option_values array for the next iteration (which is the previous time step)
            # We can slice it to keep only relevant part, but reassigning is fine.
            option_values = new_values

        return option_values[0]

if __name__ == "__main__":
    try:
        print("Trinomial Tree Option Pricer (Kamrad-Ritchken)")
        print("Please enter the following parameters:")
        
        # Default values for quick testing if user just hits enter (optional, but good for UX)
        # S=100, K=100, r=0.05, T=1, N=100, sigma=0.2
        
        S_in = input("Stock price (S) [100]: ")
        S = float(S_in) if S_in.strip() else 100.0
        
        K_in = input("Strike price (K) [100]: ")
        K = float(K_in) if K_in.strip() else 100.0
        
        r_in = input("Risk-free rate (r) [0.05]: ")
        r = float(r_in) if r_in.strip() else 0.05
        
        T_in = input("Time to maturity (T) [1.0]: ")
        T = float(T_in) if T_in.strip() else 1.0
        
        sigma_in = input("Volatility (sigma) [0.2]: ")
        sigma = float(sigma_in) if sigma_in.strip() else 0.2
        
        N_in = input("Number of steps (N) [100]: ")
        N = int(N_in) if N_in.strip() else 100
        
        pricer = TrinomialOptionPricer(S, K, T, r, sigma, N)
        
        print("\n" + "=" * 40)
        print(f"Parameters: S={S}, K={K}, r={r}, T={T}, sigma={sigma}, N={N}")
        print(f"Calculated time step (dt): {pricer.dt:.6f}")
        print("-" * 40)
        print(f"Tree Parameters:")
        print(f"u:  {pricer.u:.6f}")
        print(f"d:  {pricer.d:.6f}")
        print(f"pu: {pricer.pu:.6f}")
        print(f"pm: {pricer.pm:.6f}")
        print(f"pd: {pricer.pd:.6f}")
        print("=" * 40 + "\n")

        # Price European and American Calls and Puts
        euro_call = pricer.price('call', 'european')
        amer_call = pricer.price('call', 'american')
        euro_put = pricer.price('put', 'european')
        amer_put = pricer.price('put', 'american')

        print(f"{'Option Type':<20} | {'Price':<10}")
        print("-" * 33)
        print(f"{'European Call':<20} | {euro_call:.4f}")
        print(f"{'American Call':<20} | {amer_call:.4f}")
        print(f"{'European Put':<20} | {euro_put:.4f}")
        print(f"{'American Put':<20} | {amer_put:.4f}")
        print("-" * 33)

    except ValueError:
        print("\nInvalid input. Please ensure you enter numbers.")
    except KeyboardInterrupt:
        print("\nOperation cancelled.")
