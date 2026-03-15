import numpy as np

class CRROptionPricer:
    def __init__(self, S, K, T, r, sigma, N):
        """
        Initializes the CRR Option Pricer.

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
        self.N = N

        # Calculated parameters
        self.dt = T / N
        self.u = np.exp(sigma * np.sqrt(self.dt))
        self.d = 1 / self.u
        self.p = (np.exp(r * self.dt) - self.d) / (self.u - self.d)
        self.discount = np.exp(-r * self.dt)

    def _initialize_tree(self):
        """
        Initializes the asset prices at maturity (Step N).
        """
        asset_prices = np.zeros(self.N + 1)
        for i in range(self.N + 1):
            asset_prices[i] = self.S * (self.u ** (self.N - i)) * (self.d ** i)
        return asset_prices

    def price(self, option_type='call', style='european'):
        """
        Calculates the option price and initial delta using the CRR binomial tree model.
        
        Parameters:
        option_type (str): 'call' or 'put'
        style (str): 'european' or 'american'
        
        Returns:
        tuple: (Calculated option price, Initial Delta)
        """
        option_type = option_type.lower()
        style = style.lower()
        
        # Initialize asset prices at maturity
        asset_prices = self._initialize_tree()
        
        # Initialize option values at maturity
        option_values = np.zeros(self.N + 1)
        for i in range(self.N + 1):
            if option_type == 'call':
                option_values[i] = max(0, asset_prices[i] - self.K)
            else: # put
                option_values[i] = max(0, self.K - asset_prices[i])
        
        # Variable to store initial delta
        initial_delta = 0.0

        # Backward induction
        for j in range(self.N - 1, -1, -1):
            # Capture values for Delta calculation at the first step (t=0, j=0)
            if j == 0:
                # needed values are at t=1: V_u (index 0) and V_d (index 1)
                V_u = option_values[0]
                V_d = option_values[1]
                S_u = self.S * self.u
                S_d = self.S * self.d
                initial_delta = (V_u - V_d) / (S_u - S_d)

            for i in range(j + 1):
                # Calculate the underlying asset price at node (j, i)
                S_node = self.S * (self.u ** (j - i)) * (self.d ** i)
                
                # Continuation value
                continuation_value = self.discount * (self.p * option_values[i] + (1 - self.p) * option_values[i+1])
                
                if style == 'american':
                    # Intrinsic value calculation
                    if option_type == 'call':
                        intrinsic_value = max(0, S_node - self.K)
                    else: # put
                        intrinsic_value = max(0, self.K - S_node)
                    
                    option_values[i] = max(intrinsic_value, continuation_value)
                else:
                    option_values[i] = continuation_value
                    
        return option_values[0], initial_delta

if __name__ == "__main__":
    try:
        print("CRR Binomial Tree Option Pricer")
        print("Please enter the following parameters:")
        
        S = float(input("Stock price: "))
        K = float(input("Exercise Price: "))
        r = float(input("Risk-free rate (e.g., 0.05 for 5%): "))
        T = float(input("Time to Exercise (in years): "))
        N = int(input("Tree Steps: "))
        sigma = float(input("Volatility (e.g., 0.2 for 20%): "))
        
        pricer = CRROptionPricer(S, K, T, r, sigma, N)
        
        print("\n" + "=" * 30)
        print("Computed Tree Parameters:")
        print(f"up:           {pricer.u:.6f}")
        print(f"down:         {pricer.d:.6f}")
        print(f"probability:  {pricer.p:.6f}")
        print("=" * 30 + "\n")

        # Price European and American Calls and Puts
        euro_call, delta_euro_call = pricer.price('call', 'european')
        amer_call, delta_amer_call = pricer.price('call', 'american')
        euro_put, delta_euro_put = pricer.price('put', 'european')
        amer_put, delta_amer_put = pricer.price('put', 'american')

        def print_result(name, price, delta):
            print(f"{name}: {price:.4f}")
            print(f"  Delta: {delta:.4f}")
            action = "Buy" if delta > 0 else "Sell"
            print(f"  Replication: {action} {abs(delta):.4f} shares of stock")
            print("-" * 20)

        print("Option Prices & Deltas:")
        print("-" * 20)
        print_result("European Call", euro_call, delta_euro_call)
        print_result("American Call", amer_call, delta_amer_call)
        print_result("European Put ", euro_put, delta_euro_put)
        print_result("American Put ", amer_put, delta_amer_put)

    except ValueError:
        print("\nInvalid input. Please enter numeric values for prices/rates and integers for steps.")
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
