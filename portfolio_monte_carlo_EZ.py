import numpy as np
import pandas as pd
import yfinance as yf
import scipy.optimize as sco
import matplotlib.pyplot as plt
import datetime

class MonteCarloOptimizer:
    def __init__(self, tickers, start_date, end_date):
        """
        Initializes the optimizer by fetching data from yfinance.
        """
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        
        print(f"\n[Data] Fetching data for {', '.join(tickers)}...")
        try:
            # Download adjusted close prices
            raw_data = yf.download(tickers, start=start_date, end=end_date, auto_adjust=False)
            
            if 'Adj Close' in raw_data.columns:
                self.data = raw_data['Adj Close']
            else:
                self.data = raw_data['Close']
            
            if isinstance(self.data, pd.Series):
                self.data = self.data.to_frame()
            
            if len(tickers) > 1 and isinstance(self.data.columns, pd.MultiIndex):
                 self.data = self.data.droplevel(0, axis=1)
            
            self.data.dropna(inplace=True)
            
            if self.data.empty:
                raise ValueError("No data fetched. Check tickers or date range.")
                
            self.tickers = self.data.columns.tolist()
            
            # Calculate Daily Returns
            self.returns = self.data.pct_change().dropna()
            
            # Annualized Statistics
            self.mean_returns = self.returns.mean() * 252
            self.cov_matrix = self.returns.cov() * 252
            self.num_assets = len(self.mean_returns)
            
            print(f"[Data] Successfully loaded {self.num_assets} assets with {len(self.returns)} data points.")
            
        except Exception as e:
            print(f"[Error] Failed to fetch data: {e}")
            raise

    def portfolio_performance(self, weights):
        weights = np.array(weights)
        port_return = np.sum(self.mean_returns * weights)
        port_volatility = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
        return port_return, port_volatility

    def minimize_volatility(self):
        num_assets = self.num_assets
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0.0, 1.0) for asset in range(num_assets))
        init_guess = num_assets * [1. / num_assets,]
        result = sco.minimize(lambda w: self.portfolio_performance(w)[1],
                              init_guess, method='SLSQP', bounds=bounds, constraints=constraints)
        return result

    def maximize_sharpe(self, risk_free_rate=0.0):
        num_assets = self.num_assets
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0.0, 1.0) for asset in range(num_assets))
        init_guess = num_assets * [1. / num_assets,]
        def neg_sharpe_ratio(weights, rf):
            p_ret, p_vol = self.portfolio_performance(weights)
            return -(p_ret - rf) / p_vol
        result = sco.minimize(neg_sharpe_ratio, init_guess, args=(risk_free_rate,),
                              method='SLSQP', bounds=bounds, constraints=constraints)
        return result

    def efficient_frontier(self, points=50):
        min_vol_result = self.minimize_volatility()
        max_sharpe_result = self.maximize_sharpe() 
        ret_min_vol, _ = self.portfolio_performance(min_vol_result.x)
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
                                  init_guess, method='SLSQP', bounds=bounds, constraints=constraints)
            frontier_volatility.append(result['fun'])
            
        return target_returns, frontier_volatility

    def simulate_portfolios(self, n_portfolios=25000, risk_free_rate=0.0):
        """
        Simulates random portfolios to visualize the feasible set.
        """
        results = np.zeros((3, n_portfolios))
        for i in range(n_portfolios):
            weights = np.random.dirichlet(np.ones(self.num_assets))
            p_ret, p_vol = self.portfolio_performance(weights)
            p_sharpe = (p_ret - risk_free_rate) / p_vol
            results[0, i] = p_ret
            results[1, i] = p_vol
            results[2, i] = p_sharpe
        return results

    def plot_monte_carlo(self, risk_free_rate=0.0):
        # 1. Simulate
        print(f"[Simulation] Generating 25,000 random portfolios...")
        sim_results = self.simulate_portfolios(n_portfolios=25000, risk_free_rate=risk_free_rate)
        
        # 2. Optimize for Key Points
        min_vol_res = self.minimize_volatility()
        max_sharpe_res = self.maximize_sharpe(risk_free_rate)
        
        mv_ret, mv_vol = self.portfolio_performance(min_vol_res.x)
        ms_ret, ms_vol = self.portfolio_performance(max_sharpe_res.x)
        
        # 3. Calculate Frontier
        f_rets, f_vols = self.efficient_frontier()
        
        # Plotting
        plt.figure(figsize=(12, 8))
        
        # Cloud
        plt.scatter(sim_results[1, :], sim_results[0, :], c=sim_results[2, :], 
                    cmap='viridis', marker='o', s=5, alpha=0.5, label='Simulated Portfolios')
        plt.colorbar(label='Sharpe Ratio')
        
        # Efficient Frontier
        plt.plot(f_vols, f_rets, 'k--', linewidth=2.5, label='Efficient Frontier')
        
        # Key Points with Annotations
        plt.scatter(mv_vol, mv_ret, color='blue', s=200, marker='*', label='Min Volatility', edgecolors='white')
        plt.scatter(ms_vol, ms_ret, color='red', s=200, marker='*', label='Max Sharpe Ratio', edgecolors='white')
        
        def annotate_point(vol, ret, name):
            plt.annotate(f"{name}\n({vol:.2%}, {ret:.2%})", 
                         (vol, ret), 
                         xytext=(10, 0), 
                         textcoords='offset points',
                         fontsize=9,
                         bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.7),
                         arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))

        annotate_point(mv_vol, mv_ret, 'Min Volatility')
        annotate_point(ms_vol, ms_ret, 'Max Sharpe')
        
        # Individual Assets
        for ticker in self.tickers:
             ret = self.mean_returns[ticker]
             vol = np.sqrt(self.cov_matrix.loc[ticker, ticker])
             plt.scatter(vol, ret, s=80, edgecolors='black', label=ticker)
             plt.annotate(ticker, (vol, ret), xytext=(5, 5), textcoords='offset points', fontsize=10, fontweight='bold')

        plt.title(f'Monte Carlo Simulation & Efficient Frontier ({len(self.tickers)} Assets)')
        plt.xlabel('Annualized Volatility')
        plt.ylabel('Annualized Return')
        plt.legend(loc='best')
        plt.grid(True, linestyle='--', alpha=0.6)
        
        # Display Individual Asset Statistics
        print("\n" + "="*50)
        print(f"{'Asset':<15} | {'Ann. Return':<12} | {'Ann. Volatility':<15}")
        print("-" * 50)
        for ticker in self.tickers:
             ret = self.mean_returns[ticker]
             vol = np.sqrt(self.cov_matrix.loc[ticker, ticker])
             print(f"{ticker:<15} | {ret:.4f}       | {vol:.4f}")
        print("="*50)
        
        print(f"\nOptimization Complete.")
        print(f"Max Sharpe: {ms_ret:.2%} Ret / {ms_vol:.2%} Vol")
        print(f"Min Volatility: {mv_ret:.2%} Ret / {mv_vol:.2%} Vol")
        
        plt.show()

def get_user_inputs():
    print("Monte Carlo Portfolio Visualizer")
    print("="*40)
    
    # Get Number of Tickers
    while True:
        try:
            num_in = input("How many assets? [Default: 4]: ").strip()
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
            tick = input(f"Enter Ticker #{i+1}: ").strip().upper()
            while not tick:
                tick = input(f"Ticker cannot be empty. Enter Ticker #{i+1}: ").strip().upper()
            tickers.append(tick)
            
    # Date Range
    print("\nDate Range:")
    today = datetime.date.today()
    default_start = today - datetime.timedelta(days=365*3) # 3 years
    
    start_in = input(f"Start Date [{default_start}]: ").strip()
    end_in = input(f"End Date [{today}]: ").strip()
    
    start_date = start_in if start_in else str(default_start)
    end_date = end_in if end_in else str(today)
    
    return tickers, start_date, end_date

if __name__ == "__main__":
    try:
        tickers, start, end = get_user_inputs()
        mc_opt = MonteCarloOptimizer(tickers, start, end)
        
        # Use a fixed RF rate for Sharpe coloring, or 0.04 default
        mc_opt.plot_monte_carlo(risk_free_rate=0.04)
        
    except KeyboardInterrupt:
        print("\nOperation cancelled.")
    except Exception as e:
        print(f"\nAn error occurred: {e}")
