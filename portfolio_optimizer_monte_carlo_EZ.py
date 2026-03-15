import numpy as np
import pandas as pd
import yfinance as yf
import scipy.optimize as sco
import matplotlib.pyplot as plt
import datetime

class MeanVarianceOptimizer:
    def __init__(self, tickers, start_date, end_date):
        """
        Initializes the optimizer by fetching data from yfinance.
        """
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        
        print(f"\n[Data] Fetching data for {', '.join(tickers)}...")
        try:
            # Download adjusted close prices (force auto_adjust=False to ensure Adj Close exists,
            # or use Close if auto_adjusted. Let's use auto_adjust=False for safety to get distinct fields)
            raw_data = yf.download(tickers, start=start_date, end=end_date, auto_adjust=False)
            
            if 'Adj Close' in raw_data.columns:
                self.data = raw_data['Adj Close']
            else:
                self.data = raw_data['Close'] # Fallback
            
            # Handle single ticker case (Series -> DataFrame)
            if isinstance(self.data, pd.Series):
                self.data = self.data.to_frame()
                
            # If multiple tickers, ensure columns match (sometimes yfinance returns MultiIndex if messy)
            if len(tickers) > 1 and isinstance(self.data.columns, pd.MultiIndex):
                 self.data = self.data.droplevel(0, axis=1) # Depends on yfinance version, usually straightforward
            
            # Drop missing data
            self.data.dropna(inplace=True)
            
            if self.data.empty:
                raise ValueError("No data fetched. Check tickers or date range.")
                
            # Update tickers list to match the DataFrame column order (yfinance sorts alphabetically)
            self.tickers = self.data.columns.tolist()
            
            # Calculate Daily Returns
            self.returns = self.data.pct_change().dropna()
            
            # Annualized Statistics (assuming 252 trading days)
            self.mean_returns = self.returns.mean() * 252
            self.cov_matrix = self.returns.cov() * 252
            self.num_assets = len(self.mean_returns)
            
            print(f"[Data] Successfully loaded {self.num_assets} assets with {len(self.returns)} data points.")
            
        except Exception as e:
            print(f"[Error] Failed to fetch data: {e}")
            raise

    def portfolio_performance(self, weights):
        """
        Calculates portfolio annualized return and volatility.
        """
        weights = np.array(weights)
        port_return = np.sum(self.mean_returns * weights)
        port_volatility = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
        return port_return, port_volatility

    def minimize_volatility(self):
        """
        Finds the portfolio with Minimum Variance (Volatility).
        """
        num_assets = self.num_assets
        args = ()
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0.0, 1.0) for asset in range(num_assets))
        
        # Initial guess: Equal weights
        init_guess = num_assets * [1. / num_assets,]

        result = sco.minimize(lambda w: self.portfolio_performance(w)[1], # Min Volatility
                              init_guess,
                              method='SLSQP',
                              bounds=bounds,
                              constraints=constraints)
        
        return result

    def maximize_sharpe(self, risk_free_rate=0.0):
        """
        Finds the portfolio with Maximum Sharpe Ratio.
        """
        num_assets = self.num_assets
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0.0, 1.0) for asset in range(num_assets))
        init_guess = num_assets * [1. / num_assets,]

        def neg_sharpe_ratio(weights, risk_free_rate):
            p_ret, p_vol = self.portfolio_performance(weights)
            return -(p_ret - risk_free_rate) / p_vol

        result = sco.minimize(neg_sharpe_ratio,
                              init_guess,
                              args=(risk_free_rate,),
                              method='SLSQP',
                              bounds=bounds,
                              constraints=constraints)
        
        return result

    def equal_weights(self):
        """
        Returns performance of Equal-Weighted portfolio.
        """
        num_assets = self.num_assets
        weights = np.array([1.0/num_assets] * num_assets)
        ret, vol = self.portfolio_performance(weights)
        return weights, ret, vol

    def efficient_frontier(self, points=50):
        """
        Calculates the Efficient Frontier curve.
        """
        min_vol_result = self.minimize_volatility()
        max_sharpe_result = self.maximize_sharpe() # To find the upper bound roughly
        
        ret_min_vol, vol_min_vol = self.portfolio_performance(min_vol_result.x)
        ret_max_sharpe, vol_max_sharpe = self.portfolio_performance(max_sharpe_result.x)
        
        # Range of target returns (from min var return to max sharpe return * 1.5 for visualization)
        # Actually frontier goes up to the highest returning asset
        max_possible_return = self.mean_returns.max()
        target_returns = np.linspace(ret_min_vol, max_possible_return, points)
        
        frontier_volatility = []
        
        for target in target_returns:
            constraints = (
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
                {'type': 'eq', 'fun': lambda x, t=target: self.portfolio_performance(x)[0] - t}
            )
            bounds = tuple((0.0, 1.0) for asset in range(self.num_assets))
            init_guess = self.num_assets * [1. / self.num_assets,]
            
            result = sco.minimize(lambda w: self.portfolio_performance(w)[1],
                                  init_guess,
                                  method='SLSQP',
                                  bounds=bounds,
                                  constraints=constraints)
            
            frontier_volatility.append(result['fun'])
            
        return target_returns, frontier_volatility

    def fetch_risk_free_rate(self):
        """
        Fetches the current 13-week Treasury Bill rate (^IRX) from Yahoo Finance.
        Returns float (e.g. 0.045 for 4.5%).
        """
        try:
            print("[Data] Fetching Risk-Free Rate (^IRX)...")
            ticker = yf.Ticker("^IRX")
            # Get the most recent close
            history = ticker.history(period="1mo")
            if not history.empty:
                rate = history['Close'].iloc[-1] / 100.0
                print(f"[Data] Risk-Free Rate fetched: {rate:.2%}")
                return rate
        except Exception as e:
            print(f"[Warning] Could not fetch risk-free rate ({e}). Defaulting to 4.0%.")
        
        return 0.04

    def simulate_portfolios(self, n_portfolios=10000, risk_free_rate=0.0):
        """
        Simulates random portfolios to visualize the feasible set.
        """
        results = np.zeros((3, n_portfolios))
        
        for i in range(n_portfolios):
            # Generate random weights using Dirichlet distribution for uniform sampling on the simplex
            weights = np.random.dirichlet(np.ones(self.num_assets))
            
            p_ret, p_vol = self.portfolio_performance(weights)
            p_sharpe = (p_ret - risk_free_rate) / p_vol
            
            results[0, i] = p_ret
            results[1, i] = p_vol
            results[2, i] = p_sharpe
            
        return results

    def plot_frontier(self, risk_free_rate=None):
        """
        Plots the Efficient Frontier, optimized portfolios, Capital Market Line (CML),
        and a Monte Carlo simulation of random portfolios.
        """
        # Fetch dynamic rate if not provided
        if risk_free_rate is None:
            risk_free_rate = self.fetch_risk_free_rate()
            
        # Optimize
        min_vol_res = self.minimize_volatility()
        max_sharpe_res = self.maximize_sharpe(risk_free_rate)
        
        mv_ret, mv_vol = self.portfolio_performance(min_vol_res.x)
        ms_ret, ms_vol = self.portfolio_performance(max_sharpe_res.x)
        ew_weights, ew_ret, ew_vol = self.equal_weights()
        
        # Frontier
        f_rets, f_vols = self.efficient_frontier()
        
        # Monte Carlo Simulation
        print(f"[Simulation] Simulating 10,000 random portfolios...")
        sim_results = self.simulate_portfolios(n_portfolios=10000, risk_free_rate=risk_free_rate)
        
        plt.figure(figsize=(12, 8))
        
        # Plot Monte Carlo Cloud
        plt.scatter(sim_results[1, :], sim_results[0, :], c=sim_results[2, :], 
                    cmap='viridis', marker='o', s=10, alpha=0.3, label='Simulated Portfolios')
        plt.colorbar(label='Sharpe Ratio')

        # Plot Frontier
        plt.plot(f_vols, f_rets, 'k--', linewidth=2, label='Efficient Frontier')
        
        # Plot Capital Market Line (CML)
        # Line from (0, rf) to (ms_vol, ms_ret), extended slightly
        cml_x = [0, ms_vol, ms_vol * 1.5]
        # Equation: y = rf + (Sharpe * x)
        sharpe = (ms_ret - risk_free_rate) / ms_vol
        cml_y = [risk_free_rate, ms_ret, risk_free_rate + sharpe * (ms_vol * 1.5)]
        
        plt.plot(cml_x, cml_y, color='purple', linestyle='-.', linewidth=2, alpha=0.7, label='Capital Market Line (CML)')
        
        # Plot Portfolios with Annotations
        def annotate_point(vol, ret, name, color):
            plt.scatter(vol, ret, color=color, s=200, marker='*', label=name if name != 'Equal Weight' else name)
            if name == 'Equal Weight':
                plt.scatter(vol, ret, color=color, s=150, marker='o') # Re-draw marker if needed
            
            plt.annotate(f"{name}\n({vol:.2%}, {ret:.2%})", 
                         (vol, ret), 
                         xytext=(10, 0), 
                         textcoords='offset points',
                         fontsize=9,
                         bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="b", alpha=0.8),
                         arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))

        annotate_point(mv_vol, mv_ret, 'Min Volatility', 'blue')
        annotate_point(ms_vol, ms_ret, 'MSR / Tangency', 'red')
        annotate_point(ew_vol, ew_ret, 'Equal Weight', 'green')
        
        # Plot Individual Assets
        for ticker in self.tickers:
            idx = self.tickers.index(ticker)
            try:
                # Approximate asset mean/vol from computed stats
                # self.mean_returns is Series, self.cov_matrix is DataFrame
                asset_ret = self.mean_returns[ticker]
                asset_vol = np.sqrt(self.cov_matrix.loc[ticker, ticker])
                plt.scatter(asset_vol, asset_ret, s=50, label=ticker, alpha=0.6, edgecolors='black')
                plt.annotate(ticker, (asset_vol, asset_ret), xytext=(5, 5), textcoords='offset points', fontsize=8, fontweight='bold')
            except:
                pass

        plt.title(f'Efficient Frontier & Monte Carlo Simulation ({len(self.tickers)} Assets)\nRisk-Free Rate: {risk_free_rate:.2%}')
        plt.xlabel('Annualized Volatility')
        plt.ylabel('Annualized Return')
        plt.legend(loc='best')
        plt.grid(True, linestyle='--', alpha=0.7)
        # plt.show() removed to allow printing results first
        
        # Return results to print
        return {
            'MinVar': (min_vol_res.x, mv_ret, mv_vol),
            'MaxSharpe': (max_sharpe_res.x, ms_ret, ms_vol),
            'EqualWeight': (ew_weights, ew_ret, ew_vol),
            'RiskFreeRate': risk_free_rate
        }

def get_user_inputs():
    print("Portfolio Optimizer Configuration")
    print("="*40)
    
    # Get Number of Tickers
    while True:
        try:
            num_in = input("How many assets do you want to include? [Default: 4]: ").strip()
            if not num_in:
                num_tickers = 4
                break
            num_tickers = int(num_in)
            if num_tickers > 1:
                break
            print("Please enter a number > 1.")
        except ValueError:
            print("Invalid number.")

    tickers = []
    # If default
    if not num_in:
        print("Using default tickers: ['SPY', 'QQQ', 'AGG', 'GLD']")
        tickers = ['SPY', 'QQQ', 'AGG', 'GLD']
    else:
        for i in range(num_tickers):
            tick = input(f"Enter Ticker #{i+1} (e.g. AAPL): ").strip().upper()
            while not tick:
                tick = input(f"Ticker cannot be empty. Enter Ticker #{i+1}: ").strip().upper()
            tickers.append(tick)
            
    # Date Range
    print("\nDate Range:")
    today = datetime.date.today()
    default_start = today - datetime.timedelta(days=365*3) # 3 years
    
    start_in = input(f"Start Date (YYYY-MM-DD) [{default_start}]: ").strip()
    end_in = input(f"End Date (YYYY-MM-DD) [{today}]: ").strip()
    
    start_date = start_in if start_in else str(default_start)
    end_date = end_in if end_in else str(today)
    
    return tickers, start_date, end_date

if __name__ == "__main__":
    try:
        tickers, start, end = get_user_inputs()
        
        optimizer = MeanVarianceOptimizer(tickers, start, end)
        
        # risk_free_rate=None triggers the fetch
        results = optimizer.plot_frontier(risk_free_rate=None) 
        
        rf = results['RiskFreeRate']
        
        # Display Individual Asset Statistics
        print("\n" + "="*80)
        print(f"{'Asset':<15} | {'Ann. Return':<12} | {'Ann. Volatility':<15}")
        print("-" * 80)
        for ticker in optimizer.tickers:
             ret = optimizer.mean_returns[ticker]
             vol = np.sqrt(optimizer.cov_matrix.loc[ticker, ticker])
             print(f"{ticker:<15} | {ret:.4f}       | {vol:.4f}")
             
        # Display Table
        print("\n" + "="*80)
        print(f"Risk-Free Rate Used: {rf:.2%}")
        print(f"{'Portfolio':<15} | {'Return':<10} | {'Volatility':<10} | {'Sharpe':<10} | {'Weights'}")
        print("-" * 80)
        
        def print_row(name, data):
            weights, ret, vol = data
            sharpe = (ret - rf) / vol
            # Format weights
            w_str = ", ".join([f"{t}:{w:.2f}" for t, w in zip(optimizer.tickers, weights)])
            print(f"{name:<15} | {ret:.4f}     | {vol:.4f}     | {sharpe:.4f}     | {w_str}")

        print_row("Min Variance", results['MinVar'])
        print_row("Max Sharpe", results['MaxSharpe'])
        print_row("Equal Weight", results['EqualWeight'])
        print("="*80)
        
        # Interpretations
        print("\nanalysis & Interpretation:")
        print("-" * 30)
        
        # Max Sharpe Analysis
        ms_w, ms_ret, ms_vol = results['MaxSharpe']
        ms_sharpe = (ms_ret - rf) / ms_vol
        print(f"1. Maximum Sharpe Ratio Portfolio (Sharpe: {ms_sharpe:.2f})")
        print(f"   - This corresponds to the Tangency Portfolio on the Capital Market Line (CML).")
        print(f"   - It suggests allocating heavily to: {', '.join([f'{t} ({w:.0%})' for t, w in zip(optimizer.tickers, ms_w) if w > 0.1])}.")
        
        # Min Variance Analysis
        mv_w, mv_ret, mv_vol = results['MinVar']
        print(f"\n2. Minimum Variance Portfolio (Vol: {mv_vol:.2%})")
        print(f"   - This is the safest theoretical portfolio on the frontier.")
        
        # Comparison with EW
        ew_w, ew_ret, ew_vol = results['EqualWeight']
        ew_sharpe = (ew_ret - rf) / ew_vol
        
        improvement = (ms_sharpe - ew_sharpe) / ew_sharpe
        print(f"\n3. Optimization Gain")
        print(f"   - Optimizing for Sharpe Ratio improved efficiency by {improvement:+.1%} compared to simply holding Equal Weights.")
        print(f"   - Note: This is based on historical data ({start} to {end}). Past performance does not guarantee future results.")

        print("\nOptimization Complete. Displaying plot...")
        plt.show()
        
    except KeyboardInterrupt:
        print("\nOperation cancelled.")
    except Exception as e:
        print(f"\nAn error occurred: {e}")
