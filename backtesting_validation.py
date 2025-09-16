"""
Backtesting and Validation Module
Provides comprehensive backtesting and validation capabilities
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

class BacktestingValidator:
    """Advanced backtesting and validation system"""
    
    def __init__(self):
        self.results = {}
        self.performance_metrics = {}
        
    def backtest_strategy(self, symbol, start_date, end_date, strategy_func, initial_capital=10000):
        """Backtest a trading strategy"""
        try:
            # Get historical data
            ticker = yf.Ticker(symbol)
            hist = ticker.history(start=start_date, end=end_date)
            
            if hist.empty:
                return None
            
            # Apply strategy
            signals = strategy_func(hist)
            
            # Calculate returns
            returns = self._calculate_returns(hist, signals, initial_capital)
            
            # Calculate performance metrics
            metrics = self._calculate_performance_metrics(returns, hist)
            
            return {
                'symbol': symbol,
                'start_date': start_date,
                'end_date': end_date,
                'initial_capital': initial_capital,
                'final_capital': returns['capital'].iloc[-1],
                'total_return': (returns['capital'].iloc[-1] - initial_capital) / initial_capital,
                'returns': returns,
                'metrics': metrics,
                'signals': signals
            }
            
        except Exception as e:
            print(f"Error backtesting {symbol}: {e}")
            return None
    
    def _calculate_returns(self, hist, signals, initial_capital):
        """Calculate portfolio returns"""
        try:
            returns = pd.DataFrame(index=hist.index)
            returns['price'] = hist['Close']
            returns['signal'] = signals
            returns['position'] = 0
            returns['capital'] = initial_capital
            returns['shares'] = 0
            
            position = 0
            capital = initial_capital
            shares = 0
            
            for i in range(len(returns)):
                signal = returns['signal'].iloc[i]
                price = returns['price'].iloc[i]
                
                if signal == 1 and position == 0:  # Buy
                    shares = capital / price
                    position = 1
                    capital = 0
                elif signal == -1 and position == 1:  # Sell
                    capital = shares * price
                    shares = 0
                    position = 0
                
                returns['position'].iloc[i] = position
                returns['capital'].iloc[i] = capital + shares * price
                returns['shares'].iloc[i] = shares
            
            return returns
            
        except Exception as e:
            print(f"Error calculating returns: {e}")
            return pd.DataFrame()
    
    def _calculate_performance_metrics(self, returns, hist):
        """Calculate comprehensive performance metrics"""
        try:
            if returns.empty:
                return {}
            
            # Basic metrics
            total_return = (returns['capital'].iloc[-1] - returns['capital'].iloc[0]) / returns['capital'].iloc[0]
            annualized_return = (1 + total_return) ** (252 / len(returns)) - 1
            
            # Calculate daily returns
            daily_returns = returns['capital'].pct_change().dropna()
            
            # Volatility
            volatility = daily_returns.std() * np.sqrt(252)
            
            # Sharpe ratio
            risk_free_rate = 0.02  # 2% risk-free rate
            sharpe_ratio = (annualized_return - risk_free_rate) / volatility if volatility > 0 else 0
            
            # Maximum drawdown
            cumulative = (1 + daily_returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            max_drawdown = drawdown.min()
            
            # Sortino ratio
            negative_returns = daily_returns[daily_returns < 0]
            downside_volatility = negative_returns.std() * np.sqrt(252)
            sortino_ratio = (annualized_return - risk_free_rate) / downside_volatility if downside_volatility > 0 else 0
            
            # Calmar ratio
            calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
            
            # Win rate
            winning_days = len(daily_returns[daily_returns > 0])
            total_days = len(daily_returns)
            win_rate = winning_days / total_days if total_days > 0 else 0
            
            # Average win/loss
            avg_win = daily_returns[daily_returns > 0].mean() if len(daily_returns[daily_returns > 0]) > 0 else 0
            avg_loss = daily_returns[daily_returns < 0].mean() if len(daily_returns[daily_returns < 0]) > 0 else 0
            profit_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else 0
            
            # Value at Risk (VaR)
            var_95 = np.percentile(daily_returns, 5)
            var_99 = np.percentile(daily_returns, 1)
            
            # Expected Shortfall (ES)
            es_95 = daily_returns[daily_returns <= var_95].mean()
            es_99 = daily_returns[daily_returns <= var_99].mean()
            
            return {
                'total_return': total_return,
                'annualized_return': annualized_return,
                'volatility': volatility,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'sortino_ratio': sortino_ratio,
                'calmar_ratio': calmar_ratio,
                'win_rate': win_rate,
                'profit_loss_ratio': profit_loss_ratio,
                'var_95': var_95,
                'var_99': var_99,
                'es_95': es_95,
                'es_99': es_99
            }
            
        except Exception as e:
            print(f"Error calculating performance metrics: {e}")
            return {}
    
    def walk_forward_analysis(self, symbol, start_date, end_date, strategy_func, 
                            training_period=252, testing_period=63, step_size=21):
        """Perform walk-forward analysis"""
        try:
            results = []
            current_date = pd.to_datetime(start_date)
            end_date = pd.to_datetime(end_date)
            
            while current_date + timedelta(days=training_period + testing_period) <= end_date:
                # Training period
                train_start = current_date
                train_end = current_date + timedelta(days=training_period)
                
                # Testing period
                test_start = train_end
                test_end = test_start + timedelta(days=testing_period)
                
                # Get training data
                ticker = yf.Ticker(symbol)
                train_data = ticker.history(start=train_start, end=train_end)
                test_data = ticker.history(start=test_start, end=test_end)
                
                if train_data.empty or test_data.empty:
                    current_date += timedelta(days=step_size)
                    continue
                
                # Apply strategy to test data
                test_signals = strategy_func(test_data)
                test_returns = self._calculate_returns(test_data, test_signals, 10000)
                test_metrics = self._calculate_performance_metrics(test_returns, test_data)
                
                results.append({
                    'train_start': train_start,
                    'train_end': train_end,
                    'test_start': test_start,
                    'test_end': test_end,
                    'test_returns': test_returns,
                    'test_metrics': test_metrics
                })
                
                current_date += timedelta(days=step_size)
            
            return results
            
        except Exception as e:
            print(f"Error in walk-forward analysis: {e}")
            return []
    
    def monte_carlo_simulation(self, returns, num_simulations=1000, days=252):
        """Perform Monte Carlo simulation"""
        try:
            if returns.empty:
                return {}
            
            daily_returns = returns['capital'].pct_change().dropna()
            mean_return = daily_returns.mean()
            std_return = daily_returns.std()
            
            simulations = []
            for _ in range(num_simulations):
                # Generate random returns
                random_returns = np.random.normal(mean_return, std_return, days)
                # Calculate cumulative returns
                cumulative_returns = (1 + random_returns).cumprod()
                simulations.append(cumulative_returns)
            
            simulations = np.array(simulations)
            
            # Calculate statistics
            final_values = simulations[:, -1]
            mean_final_value = np.mean(final_values)
            std_final_value = np.std(final_values)
            
            # Percentiles
            percentiles = np.percentile(final_values, [5, 10, 25, 50, 75, 90, 95])
            
            return {
                'simulations': simulations,
                'mean_final_value': mean_final_value,
                'std_final_value': std_final_value,
                'percentiles': percentiles,
                'probability_of_loss': np.mean(final_values < 1.0)
            }
            
        except Exception as e:
            print(f"Error in Monte Carlo simulation: {e}")
            return {}
    
    def stress_test(self, returns, stress_scenarios):
        """Perform stress testing"""
        try:
            if returns.empty:
                return {}
            
            daily_returns = returns['capital'].pct_change().dropna()
            initial_capital = returns['capital'].iloc[0]
            
            stress_results = {}
            
            for scenario_name, stress_factor in stress_scenarios.items():
                # Apply stress factor to returns
                stressed_returns = daily_returns * stress_factor
                stressed_capital = initial_capital * (1 + stressed_returns).cumprod()
                
                # Calculate metrics
                final_capital = stressed_capital.iloc[-1]
                total_return = (final_capital - initial_capital) / initial_capital
                max_drawdown = self._calculate_max_drawdown(stressed_capital)
                
                stress_results[scenario_name] = {
                    'final_capital': final_capital,
                    'total_return': total_return,
                    'max_drawdown': max_drawdown,
                    'stress_factor': stress_factor
                }
            
            return stress_results
            
        except Exception as e:
            print(f"Error in stress testing: {e}")
            return {}
    
    def _calculate_max_drawdown(self, capital_series):
        """Calculate maximum drawdown"""
        try:
            cumulative = capital_series / capital_series.iloc[0]
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            return drawdown.min()
        except Exception as e:
            return 0
    
    def create_performance_report(self, backtest_results):
        """Create comprehensive performance report"""
        try:
            if not backtest_results:
                return "No backtest results available"
            
            report = f"""
# Performance Report

## Summary
- **Symbol**: {backtest_results['symbol']}
- **Period**: {backtest_results['start_date']} to {backtest_results['end_date']}
- **Initial Capital**: ${backtest_results['initial_capital']:,.2f}
- **Final Capital**: ${backtest_results['final_capital']:,.2f}
- **Total Return**: {backtest_results['total_return']:.2%}

## Performance Metrics
- **Annualized Return**: {backtest_results['metrics'].get('annualized_return', 0):.2%}
- **Volatility**: {backtest_results['metrics'].get('volatility', 0):.2%}
- **Sharpe Ratio**: {backtest_results['metrics'].get('sharpe_ratio', 0):.2f}
- **Maximum Drawdown**: {backtest_results['metrics'].get('max_drawdown', 0):.2%}
- **Sortino Ratio**: {backtest_results['metrics'].get('sortino_ratio', 0):.2f}
- **Calmar Ratio**: {backtest_results['metrics'].get('calmar_ratio', 0):.2f}
- **Win Rate**: {backtest_results['metrics'].get('win_rate', 0):.2%}
- **Profit/Loss Ratio**: {backtest_results['metrics'].get('profit_loss_ratio', 0):.2f}

## Risk Metrics
- **VaR (95%)**: {backtest_results['metrics'].get('var_95', 0):.2%}
- **VaR (99%)**: {backtest_results['metrics'].get('var_99', 0):.2%}
- **Expected Shortfall (95%)**: {backtest_results['metrics'].get('es_95', 0):.2%}
- **Expected Shortfall (99%)**: {backtest_results['metrics'].get('es_99', 0):.2%}
"""
            
            return report
            
        except Exception as e:
            return f"Error creating performance report: {e}"
    
    def plot_performance(self, backtest_results):
        """Create performance visualization"""
        try:
            if not backtest_results or backtest_results['returns'].empty:
                return None
            
            returns = backtest_results['returns']
            
            # Create subplots
            fig = go.Figure()
            
            # Add capital curve
            fig.add_trace(go.Scatter(
                x=returns.index,
                y=returns['capital'],
                mode='lines',
                name='Portfolio Value',
                line=dict(color='blue', width=2)
            ))
            
            # Add price curve
            fig.add_trace(go.Scatter(
                x=returns.index,
                y=returns['price'],
                mode='lines',
                name='Stock Price',
                line=dict(color='red', width=1),
                yaxis='y2'
            ))
            
            # Update layout
            fig.update_layout(
                title=f"Performance Analysis - {backtest_results['symbol']}",
                xaxis_title="Date",
                yaxis_title="Portfolio Value ($)",
                yaxis2=dict(
                    title="Stock Price ($)",
                    overlaying="y",
                    side="right"
                ),
                hovermode='x unified'
            )
            
            return fig
            
        except Exception as e:
            print(f"Error creating performance plot: {e}")
            return None
