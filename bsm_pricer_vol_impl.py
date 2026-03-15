import numpy as np
from scipy.stats import norm

class BSMOptionPricer:
    def __init__(self, S, K, T, r, sigma, q=0.0):
        """
        Initializes the Black-Scholes-Merton Option Pricer.

        Parameters:
        S (float): Spot price
        K (float): Strike price
        T (float): Time to maturity (in years)
        r (float): Risk-free interest rate (annual)
        sigma (float): Volatility (annual)
        q (float): Continuous dividend yield (annual)
        """
        self.S = float(S)
        self.K = float(K)
        self.T = float(T)
        self.r = float(r)
        self.sigma = float(sigma)
        self.q = float(q)

    def _d1_d2(self):
        """
        Calculates d1 and d2 parameters.
        Returns None, None if T <= 0 or sigma <= 0.
        """
        if self.T <= 0 or self.sigma <= 0:
            return None, None
            
        d1 = (np.log(self.S / self.K) + (self.r - self.q + 0.5 * self.sigma ** 2) * self.T) / (self.sigma * np.sqrt(self.T))
        d2 = d1 - self.sigma * np.sqrt(self.T)
        return d1, d2

    def price(self, option_type='call'):
        """
        Calculates the option price using the BSM formula.
        Handles edge cases for T=0 and sigma=0.
        """
        option_type = option_type.lower()
        
        # Edge Case: Expiry (T <= 0)
        if self.T <= 0:
            if option_type == 'call':
                return max(0.0, self.S - self.K)
            else:
                return max(0.0, self.K - self.S)
                
        # Edge Case: Zero Volatility (sigma <= 0)
        if self.sigma <= 0:
            # Deterministic forward price: F = S * exp((r - q) * T)
            forward_price = self.S * np.exp((self.r - self.q) * self.T)
            discount_factor = np.exp(-self.r * self.T)
            
            if option_type == 'call':
                return discount_factor * max(0.0, forward_price - self.K)
            else:
                return discount_factor * max(0.0, self.K - forward_price)

        d1, d2 = self._d1_d2()
        
        # Standard BSM Formula with Dividends
        if option_type == 'call':
            price = self.S * np.exp(-self.q * self.T) * norm.cdf(d1) - self.K * np.exp(-self.r * self.T) * norm.cdf(d2)
        else: # put
            price = self.K * np.exp(-self.r * self.T) * norm.cdf(-d2) - self.S * np.exp(-self.q * self.T) * norm.cdf(-d1)
            
        return price

    def greeks(self, option_type='call'):
        """
        Calculates the Greeks for the option.
        """
        option_type = option_type.lower()
        d1, d2 = self._d1_d2()
        
        # Handle edge cases (T <= 0 or sigma <= 0) roughly for Greeks
        if d1 is None:
            # For simplicity, return 0 or approx delta strictly based on intrinsic moneyness
            return {
                'Delta': 0.0, 'Gamma': 0.0, 'Vega': 0.0, 'Theta': 0.0, 'Rho': 0.0
            }

        sqt_T = np.sqrt(self.T)
        exp_neg_qT = np.exp(-self.q * self.T)
        exp_neg_rT = np.exp(-self.r * self.T)
        N_prime_d1 = norm.pdf(d1)
        
        # Common Greeks (Math / Standard)
        gamma = (exp_neg_qT * N_prime_d1) / (self.S * self.sigma * sqt_T)
        vega_math = self.S * exp_neg_qT * N_prime_d1 * sqt_T  # Sensitivity to 1.0 change in sigma (100% vol)

        if option_type == 'call':
            delta = exp_neg_qT * norm.cdf(d1)
            
            theta_math = (- (self.S * self.sigma * exp_neg_qT * N_prime_d1) / (2 * sqt_T)
                     - self.r * self.K * exp_neg_rT * norm.cdf(d2)
                     + self.q * self.S * exp_neg_qT * norm.cdf(d1)) # Per year
                     
            rho_math = self.K * self.T * exp_neg_rT * norm.cdf(d2) # Sensitivity to 1.0 change in r (100% rate)
        else: # put
            delta = exp_neg_qT * (norm.cdf(d1) - 1)
            
            theta_math = (- (self.S * self.sigma * exp_neg_qT * N_prime_d1) / (2 * sqt_T)
                     + self.r * self.K * exp_neg_rT * norm.cdf(-d2)
                     - self.q * self.S * exp_neg_qT * norm.cdf(-d1)) # Per year
                     
            rho_math = -self.K * self.T * exp_neg_rT * norm.cdf(-d2) # Sensitivity to 1.0 change in r (100% rate)

        # Trader Greeks (scaled)
        vega_1pct = vega_math / 100.0       # Per 1% vol change
        rho_1pct = rho_math / 100.0         # Per 1% rate change
        theta_1day = theta_math / 365.0     # Per 1 day decay

        return {
            'Delta': delta,
            'Gamma': gamma,
            'Vega (Math)': vega_math,
            'Vega (1%)': vega_1pct,
            'Theta (Year)': theta_math,
            'Theta (Day)': theta_1day,
            'Rho (Math)': rho_math,
            'Rho (1%)': rho_1pct
        }

    def implied_volatility(self, market_price, option_type='call', tol=1e-5, max_iter=100):
        """
        Calculates the Implied Volatility using Newton-Raphson method.
        
        Parameters:
        market_price (float): The observed market price of the option
        option_type (str): 'call' or 'put'
        tol (float): Tolerance for convergence
        max_iter (int): Maximum number of iterations
        
        Returns:
        float: Implied Volatility (sigma)
        """
        option_type = option_type.lower()
        original_sigma = self.sigma
        
        # Initial guess
        self.sigma = 0.5 # Start with 50% vol
        
        for i in range(max_iter):
            price = self.price(option_type)
            diff = market_price - price
            
            if abs(diff) < tol:
                iv = self.sigma
                self.sigma = original_sigma # Restore
                return iv
                
            greeks = self.greeks(option_type)
            vega = greeks['Vega (Math)']
            
            if abs(vega) < 1e-8:
                # Vega too small (option likely far ITM/OTM), Newton method fails
                # Should fallback or break. For now, break.
                break
                
            self.sigma = self.sigma + diff / vega
            
        # Restore and raise error if not found
        self.sigma = original_sigma
        raise ValueError("Implied Volatility did not converge. (Check if price is within arbitrage bounds)")

if __name__ == "__main__":
    try:
        print("Black-Scholes-Merton Option Pricer (Price & Implied Volatility)")
        print("=" * 60)
        
        print("Select Mode:")
        print("  [1] Calculate Option Price (from volatility)")
        print("  [2] Calculate Implied Volatility (from market price)")
        
        mode_in = input("Enter mode [1]: ").strip()
        mode = int(mode_in) if mode_in else 1
        
        print("\nPlease enter the following parameters:")
        S = float(input("Stock price (S) [100]: ") or 100)
        K = float(input("Strike price (K) [100]: ") or 100)
        r = float(input("Risk-free rate (r) [0.05]: ") or 0.05)
        q = float(input("Dividend yield (q) [0.0]: ") or 0.0)
        T = float(input("Time to maturity (T) [1.0]: ") or 1.0)
        
        if mode == 1:
            sigma = float(input("Volatility (sigma) [0.2]: ") or 0.2)
            pricer = BSMOptionPricer(S, K, T, r, sigma, q)
            
            print("\n" + "=" * 50)
            print(f"Parameters: S={S}, K={K}, r={r}, q={q}, T={T}, sigma={sigma:.4f}")
            print("=" * 50)
            
            # Loop for both Call and Put
            types_to_run = ['call', 'put']

        elif mode == 2:
            market_price = float(input("Option Market Price: "))
            option_type_in = input("Option Type (call/put) [call]: ").lower() or 'call'
            
            # Init pricer with dummy sigma
            pricer = BSMOptionPricer(S, K, T, r, 0.2, q)
            
            try:
                iv = pricer.implied_volatility(market_price, option_type_in)
                pricer.sigma = iv # Set the pricer to the calculated IV to compute Greeks
                
                print("\n" + "=" * 50)
                print(f"Result for {option_type_in.upper()} option at price {market_price}:")
                print(f"IMPLIED VOLATILITY: {iv:.6f} ({iv*100:.2f}%)")
                print("=" * 50)
                
                # Run only for the specified type since IV is specific to that option's price
                types_to_run = [option_type_in]
                
            except ValueError as e:
                print(f"\nError: {e}")
                types_to_run = []
        
        # Display results (Price & Greeks) for selected types
        for option_type in types_to_run:
            price = pricer.price(option_type)
            greeks = pricer.greeks(option_type)
            
            print(f"\n{option_type.upper()} Option (at sigma={pricer.sigma:.4f}):")
            print("-" * 35)
            print(f"Price: {price:.4f}")
            print("Greeks:")
            order = [
                'Delta', 'Gamma', 
                'Vega (Math)', 'Vega (1%)', 
                'Theta (Year)', 'Theta (Day)', 
                'Rho (Math)', 'Rho (1%)'
            ]
            for name in order:
                val = greeks.get(name, 0.0)
                print(f"  {name:<15}: {val:.4f}")

    except ValueError:
        print("\nInvalid input. Please enter numeric values where expected.")
    except KeyboardInterrupt:
        print("\nOperation cancelled.")
