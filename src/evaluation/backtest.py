import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO

# Add the project root directory to the system path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(project_root)

# Import the environment
from src.environment.trading_env import TTLTradingEnv

# Constants
CAPITAL = 1_000_000  # Initial capital
COMMISSION = 0.00125  # 0.125% commission rate

def run_backtest(
    model_path="models/best_model/best_model",
    ticker="AMZN",
    data_path="data/processed/AMZN_features.csv",
    results_dir="results",
    ttl=10,
    window_size=1,
    no_op_penalty=-0.001,
    no_op_mod=3,
    initial_capital=CAPITAL,
    commission=COMMISSION,
    verbose=True,
    position_size_percent=10, 
    save=True
):
    """
    Run a backtest of the trading strategy.
    
    Args:
        model_path: Path to the trained model
        data_path: Path to the processed dataset
        results_dir: Directory to save results
        ttl: Time-to-live for trading decisions
        window_size: Observation window size
        no_op_penalty: Penalty for no-op actions
        no_op_mod: Apply penalty after this many no-ops
        n_shares: Number of shares to trade
        initial_capital: Initial capital for backtesting
        commission: Commission rate for trading
        verbose: Whether to print detailed information
    
    Returns:
        dict: Performance metrics
    """
    # Create results directories
    figures_dir = os.path.join(results_dir, "figures")
    trade_history_dir = os.path.join(results_dir, "trade_history")
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(trade_history_dir, exist_ok=True)
    
    # Load the dataset
    if verbose:
        print("Loading dataset from {}...".format(data_path))
    df = pd.read_csv(data_path)
    
    # Extract prices and features
    prices = df['Close'].astype(np.float32).values
    
    # Handle case where 'Date' may or may not be in columns
    drop_cols = ['Close']
    if 'Date' in df.columns:
        dates = pd.to_datetime(df['Date'])
        drop_cols.append('Date')
    else:
        dates = pd.date_range(start='2000-01-01', periods=len(prices), freq='D')
    
    features = df.drop(columns=drop_cols).astype(np.float32).values
    
    # Load the trained model
    if verbose:
        print("Loading model from {}...".format(model_path))
    model = PPO.load(model_path)
    
    # Create a test environment
    env = TTLTradingEnv(
        features=features,
        prices=prices,
        window_size=window_size,
        ttl=ttl,
        no_op_penalty=no_op_penalty,
        no_op_mod=no_op_mod
    )
    
    # Initialize variables for backtesting
    portfolio = [initial_capital]  # Start with initial capital
    capital = initial_capital
    
    # Track buy/sell points for visualization
    buy_indices = []
    buy_prices = []
    buy_dates = []
    sell_indices = []
    sell_prices = []
    sell_dates = []
    trade_history = []
    
    # Run the backtest
    if verbose:
        print("Running backtest...")
    obs = env.reset()
    done = False
    step = 0
    
    while not done:
        # Get model's action
        action, _ = model.predict(obs, deterministic=True)
        
        # Record the current index (start of the TTL period)
        current_idx = env.current_idx
        
        # Start and end days of the TTL period
        start_day = current_idx
        end_day = start_day + env.ttl - 1
        
        # Check index bounds
        if start_day >= len(prices) or end_day >= len(prices):
            if verbose:
                print("Reached end of data. start_day={}, end_day={}, data_length={}".format(start_day, end_day, len(prices)))
            break
        
        # Get prices at start and end of period
        start_price = prices[start_day]
        end_price = prices[end_day]
        
        # Get dates safely
        start_date = dates[start_day] if start_day < len(dates) else None
        end_date = dates[end_day] if end_day < len(dates) else None
        
        # Buy action
        if action == 1:
            
            # Calculate number of shares to buy
            max_capital_to_use = capital * position_size_percent / 100
            shares_to_buy = int(max_capital_to_use / start_price)
            
            if shares_to_buy < 1:
                shares_to_buy = 1  # Ensure at least 1 share
            
            # Calculate cost with commission
            cost = shares_to_buy * start_price
            commission_fee = cost * commission
            total_cost = cost + commission_fee
            
            # Update capital
            capital -= total_cost
            
            # Record the trade
            buy_indices.append(start_day)
            buy_prices.append(start_price)
            buy_dates.append(start_date)
            if start_date is not None:
                buy_dates.append(start_date)
            
            if verbose:
                print("Step {}: BUY {} shares at ${:.2f} ({}), Commission: ${:.2f}".format(step, shares_to_buy, start_price, start_date, commission_fee))
            
            # Since we just bought, we'll sell at the end of this TTL period
            # Calculate proceeds with commission
            proceeds = shares_to_buy * end_price
            commission_fee = proceeds * commission
            net_proceeds = proceeds - commission_fee
            
            # Update capital
            capital += net_proceeds
            
            # Calculate profit/loss
            pl = net_proceeds - total_cost
            percent_return = (pl / total_cost) * 100
            
            # Record the trade
            sell_indices.append(end_day)
            sell_prices.append(end_price)
            if end_date is not None:
                sell_dates.append(end_date)
            
            # Add to trade history
            trade_history.append({
                'step': step,
                'buy_date': start_date,
                'buy_day': start_day,
                'buy_price': start_price,
                'sell_date': end_date,
                'sell_day': end_day,
                'sell_price': end_price,
                'shares': shares_to_buy,
                'profit_loss': pl,
                'return_percent': percent_return
            })
            
            if verbose:
                print("Step {}: SELL {} shares at ${:.2f} ({}), Commission: ${:.2f}".format(step, shares_to_buy, end_price, end_date, commission_fee))
                print("     P/L for this instance: ${:.2f} ({:.2f}%)".format(pl, percent_return))
                print("     Current porfolio value: ${:.2f}".format(capital))
            
        
        # Update portfolio value
        portfolio.append(capital)
        
        # Take action in environment (this advances time by TTL days)
        obs, reward, done, info = env.step(action)
        
        step += 1
    
    # Calculate performance metrics
    final_capital = portfolio[-1]
    total_return = (final_capital - initial_capital) / initial_capital * 100
    total_profit = final_capital - initial_capital
    num_trades = len(buy_indices)
    
    # Calculate annualized metrics
    if len(dates) > 1 and num_trades > 0:
        # If dates are trading days
        total_days = len(prices)
        if total_days > 0:
            annualized_return = total_return * (252 / total_days)
        else:
            annualized_return = 0
    else:
        annualized_return = 0
    
    # Win rate
    winning_trades = sum(1 for trade in trade_history if trade['profit_loss'] > 0)
    win_rate = winning_trades / num_trades if num_trades > 0 else 0
    
    # Print results
    if verbose:
        print("\n===== Backtest Results =====")
        print("Initial Capital: ${:,.2f}".format(initial_capital))
        print("Final Capital: ${:,.2f}".format(final_capital))
        print("Total Return: {:.2f}%".format(total_return))
        print("Annualized Return: {:.2f}%".format(annualized_return))
        print("Total Profit/Loss: ${:,.2f}".format(total_profit))
        print("Number of Trades: {}".format(num_trades))
        print("Win Rate: {:.2f}%".format(win_rate * 100))
        print("============================")
    
        # Plot results
        plt.figure(figsize=(15, 10))
        
        # Plot portfolio value
        plt.subplot(2, 1, 1)
        plt.plot(portfolio)
        plt.title('Portfolio Value Over Time')
        plt.ylabel('Portfolio Value ($)')
        plt.grid(True)
        
        # Plot price with buy/sell points
        plt.subplot(2, 1, 2)
        plt.plot(prices)
        
        # Add buy/sell markers
        plt.scatter(buy_indices, buy_prices, color='green', marker='^', label='Buy')
        plt.scatter(sell_indices, sell_prices, color='red', marker='v', label='Sell')
        
        plt.title('{} Price with Buy/Sell Points'.format(ticker))
        plt.xlabel('Day')
        plt.ylabel('Price ($)')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
    
    # Save figure
    if save:
        figure_path = os.path.join(figures_dir, "backtest_results.png")
        plt.savefig(figure_path)
        if verbose:
            print("Saved figure to {}".format(figure_path))
            plt.show()
    
    # Save trade history
    if trade_history and save:
        trade_df = pd.DataFrame(trade_history)
        trade_history_path = os.path.join(trade_history_dir, "trade_history.csv")
        trade_df.to_csv(trade_history_path, index=False)
        if verbose:
            print("Saved trade history to {}".format(trade_history_path))
    
    # Return performance metrics
    return {
        'initial_capital': initial_capital,
        'final_capital': final_capital,
        'total_return': total_return,
        'annualized_return': annualized_return,
        'total_profit': total_profit,
        'num_trades': num_trades,
        'win_rate': win_rate
    }

if __name__ == "__main__":
    results = run_backtest(model_path="models/best_model/best_model")
    
    ITERATIONS = 100
    results = 0
    for i in range(ITERATIONS):
        result = run_backtest(model_path="models/best_model/best_model", verbose=False, save = False)
        results += result['final_capital']
    results /= ITERATIONS
    print("Average final capital over {} runs: ${:,.2f}".format(ITERATIONS, results))    
    
    
    