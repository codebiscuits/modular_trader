# Modular Trader - Cryptocurrency Algorithmic Trading System
This is a sophisticated automated cryptocurrency trading system that uses machine learning to identify and execute trading opportunities on crypto markets (primarily Binance).
## Core Functionality
The system implements a complete ML-based trading pipeline:
### Trading Strategies Implemented:
- Channel Run Strategy: Trades based on price channels (support/resistance) with configurable entry points (edge or mid-channel)
- Trail Fractals Strategy: Uses fractal patterns to identify entry and exit points
### Three-Model ML Architecture:
- Primary (Technical) Model: Predicts trade opportunities based on technical indicators from OHLC (Open/High/Low/Close) price data
- Risk Model: Evaluates the risk of potential trades using spread data, market rankings and other risk factors
- Performance Model: (Mentioned but less detailed) Tracks strategy performance over time
### Real, Sim and Tracked Trades
Since backtesting is very difficult to do without various biases creeping in, I want to train the models partly on the outcomes of their own decisions in production. To achieve this, the system uses a concept of real and simulated trades to record and track the performance of the trades that were not taken as well as the ones that were. By tracking both, the models can use this data to learn from every setup generated, not just the ones that were taken.
I extended this idea to create another trade type which I called 'tracked'. These are trades that were closed according to strategy rules rather than a stop-loss. I wanted to keep tracking these trades until such time that they would hit their stop-loss, so that I could analyse whether the strategy was closing them at the optimal time.
### Other Key Terminology
- Session: the class that contains all of the data and functionality needed to manage each trading session. Tracks and handles global risk metrics, market data, exchange information such as API limits and account balances, and launches Agents.
- Agent: the class that manages a single trading strategy. It loads a specific set of models, analyses market data according to the strategy rules, executes trades and manages its own records.
- Open Risk: represents how much of the value of the account would be lost if all currently open trades were to hit their stop-loss. The Session object monitors this metric and prevents new positions from being opened if it is too high.
- Live: This is a flag which tells the system whether it is running in dev or production mode. Since I did all dev work on one machine and ran it for real on a raspberry pi, I set it up to automatically run in live mode on the raspberry pi and in dev mode otherwise. This way, I could test almost all of the code without placing actual trades.
## Key Technical Features
- Feature Engineering: Adds 100+ technical indicators (RSI, MACD, Bollinger Bands, volume metrics, etc.)
- Spread Analysis: Incorporates bid-ask spread data to assess trading costs and estimate liquidity
- Risk Management: Built-in stop-loss and take-profit logic with risk-reward ratios
- Performance Tracking: Stores trade records and analyzes historical performance to inform how much capital to allocate to each model
- Walk-Forward Testing: Unfinished attempt to improve backtesting of models
## Workflow
- Run 'update_ohlc.py' or 'async_update_ohlc.py' periodically to update market data from exchanges. It's best to run this right before 'setup_scanner.py' to minimise the amount of data that needs to be downloaded during the trading session.
- Run 'ml_model_training.py' once per day/week to train models using recent historical data.
- Run 'setup_scanner.py' as often as needed to scan the market for setups and execute trades. How often this is run will depend on the timeframe of the models that are being used, 1d models will need to be run daily, 4h models will need to be run every 4 hours etc. The session object that manages everything has a method called 'get_timeframes' that decides which agents need to run based on what time of day it is. This means that the script can be scheduled to run every hour or even every 15 mins and each model will only run as often as they should.
## Important Notes
- Uses Binance API for market data and order execution
- Supports both live trading and simulation modes
- Includes fee calculation (0.15% estimated per trade)
- Makes use of dummy files to determine if it is running on dev machine or in production.
- Stores models in /machine_learning/ directory
- Tracks all trades in JSON files for later analysis
## Main Scripts
### setup_scanner.py
This is the "deployment script" that configures and launches trading agents with trained ML models. Think of it as the "mission control" that sets up the actual trading bots.
#### What It Does:
1. Loads Trained ML Models:
   - Reads saved Random Forest models from /machine_learning/ directory
   - Loads three types of models for each trading strategy:
     - Primary/Technical model (predicts entry signals)
     - Risk model (filters risky trades)
     - Performance model (tracks strategy effectiveness)
2. Creates Agent Configurations:
   - Defines which trading strategies to run (Channel Run, Trail Fractals)
   - Sets parameters for each strategy:
     - Timeframe (15m, 30m, 1h, 4h, 12h, 1d)
     - Direction (long/short)
     - Strategy parameters (lookback periods, entry triggers, etc.)
   - Links each agent to its trained ML models
3. Generates Agent Objects:
   - Creates Agent instances that will actually execute trades
   - Each agent represents one strategy configuration (e.g., "Channel Run Long 1h")
   - Agents contain:
     - Strategy logic
     - Risk management rules
     - Position sizing
     - ML model references
4. Saves Configuration:
   - Writes agent list to /machine_learning/agents.pkl (pickle file)
   - This allows other scripts to load the configured agents
#### Key Configuration Parameters:
```python
# Example agent configuration structure:
{
    'strat_name': 'channel_run',
    'side': 'long', 
    'timeframe': '1h',
    'strat_params': (200, 'mid', 'edge'),  # lookback, entry, goal
    'selection_method': 'volume_1d',
    'num_pairs': 100,
    'model_id': 1234567890  # timestamp of when model was trained
}
```

#### Important Notes:
- Must be run after ml_model_training.py trains new models
- The script includes hardcoded model IDs that need to be updated when retraining
- Different agents can trade the same pair if they use different timeframes/strategies
- The file has a reminder comment: "DONT FORGET TO CHANGE THE AGENT PARAMS IN SETUP SCANNER" (this appears in ml_model_training.py output)
### ml_model_training.py
This is the main training script that:
1. Generates Training Data by backtesting strategies on historical price data:
   - Loads OHLC data for multiple cryptocurrency pairs
   - Applies technical indicators (ATR, channels, moving averages, etc.)
   - Simulates trades using the strategy rules
   - Records outcomes (profit/loss, win/loss)
2. Trains Machine Learning Models using Random Forest classifiers:
   - Splits data into train/test/validation sets
   - Performs feature engineering and selection (mutual information, sequential feature selection, permutation importance)
   - Handles class imbalance using undersampling techniques
   - Optimizes hyperparameters using Optuna
   - Validates model performance
3. Saves Trained Models for use in live trading
4. Trains Models for Multiple Configurations:
   - Different timeframes (15m, 30m, 1h, 4h, 12h, 1d)
   - Long and short positions
   - Different strategy parameters
### update_ohlc.py
This script is the "data pipeline" that fetches and maintains historical price data from cryptocurrency exchanges.
#### What It Does:
1. Downloads OHLC Data from Binance:
   - Fetches Open/High/Low/Close/Volume candlestick data
   - Supports multiple timeframes (1m, 5m, 15m, 30m, 1h, 4h, 12h, 1d)
   - Downloads data for all tradeable cryptocurrency pairs on Binance
2. Data Storage:
   - Saves data as Parquet files (efficient columnar format) in /market_data/{timeframe}/{pair}.parquet
   - Example: /market_data/1h/BTCUSDT.parquet
3. Incremental Updates:
   - Checks existing data and only downloads new candles since last update
   - Avoids re-downloading historical data (saves API calls and time)
   - Handles gaps in data if the system was offline
4. Data Quality Management:
   - Validates data completeness
   - Handles missing candles
   - Ensures timestamps are correct
   - Removes duplicates
5. Multi-Timeframe Support:
   - Can update multiple timeframes in one run
   - Typically starts with 5-minute data and resamples to higher timeframes
   - Or downloads each timeframe directly from Binance
#### Key Functions:
- get_binance_ohlc(pair, timeframe, start_time, end_time)
  - Makes API request to Binance
  - Returns DataFrame with OHLC data
- update_single_pair(pair, timeframe)
  - Loads existing parquet file (if exists)
  - Determines what new data is needed
  - Fetches and appends new candles
  - Saves updated file
- update_all_pairs(timeframe)
  - Loops through all pairs
  - Updates each one
  - Handles rate limiting
  - Logs progress
#### Related File: async_update_ohlc.py
This is the asynchronous version that updates data much faster:
- Uses asyncio and aiohttp for concurrent requests
- Can update 100+ pairs simultaneously
- Same functionality, just much faster (10-20x speedup)
- Recommended for daily data updates
