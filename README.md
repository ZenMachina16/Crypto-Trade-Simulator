
# Crypto Trading Simulator: Technical Documentation

## 1. Model Choices and Equations

### 1.1 Order Book Model

The simulator uses a dual SortedDict implementation for efficient price level management:

```python
# Bids: Descending sort (highest price first)
self.bids = SortedDict(lambda k: -float(k))

# Asks: Ascending sort (lowest price first)
self.asks = SortedDict()
```

This allows O(log n) operations for insertions, deletions, and retrievals while maintaining the sorted nature of the order book.

### 1.2 Slippage Estimation Models

The simulator implements multiple regression techniques for slippage estimation:

#### Linear Regression Model
Uses polynomial features of degree 2:

```
Slippage = β₀ + β₁×x₁ + β₂×x₂ + ... + β₍ₙ₎×x₍ₙ₎ + β₍ₙ₊₁₎×x₁² + β₍ₙ₊₂₎×x₁×x₂ + ...
```

Where features include:
- Best price
- Total quantity in orderbook
- Weighted average price
- Price range
- Liquidity metrics
- Order quantity

#### Quantile Regression Model
Estimates conditional quantiles (25th, 50th, 75th percentiles) using the formula:

```
min_β Σ ρτ(y - X·β)
```

Where:
- ρτ(u) = u×(τ - I(u < 0))
- τ = desired quantile (0.25, 0.5, 0.75)
- I() = indicator function

### 1.3 Market Impact Model (Almgren-Chriss)

The Almgren-Chriss model divides market impact into temporary and permanent components:

#### Temporary Impact
```
I_temp = κ × σ × (q/V)^γ × P
```

Where:
- κ = temporary impact factor
- σ = volatility
- q = order quantity
- V = market volume
- γ = impact exponent (typically 0.5-1.5)
- P = current price

#### Permanent Impact
```
I_perm = η × (q/V) × P
```

Where:
- η = permanent impact factor
- q = order quantity
- V = market volume
- P = current price

### 1.4 Maker/Taker Probability Model

Logistic regression model to estimate the probability of an order being a maker:

```
P(maker) = 1 / (1 + e^(-z))
where z = β₀ + β₁×imbalance + β₂×bid_ask_ratio + β₃×spread + ...
```

Features include:
- Order book imbalance
- Bid/ask ratio
- Spread (absolute and percentage)
- Order quantity
- Relative size metrics

## 2. Regression Training

### 2.1 Slippage Model Training

#### Feature Extraction
```python
# Key features derived from orderbook
total_qty = sum(qty for _, qty in levels)
weighted_price = sum(price * qty for price, qty in levels) / total_qty
price_range = max(price) - min(price)
liquidity_within_1pct = sum(qty where (price - best_price)/best_price <= 0.01)
```

#### Linear Model Training
1. Extract features from historical orderbooks
2. Process trade data to calculate actual slippage
3. Generate polynomial features of degree 2
4. Train using sklearn's LinearRegression
5. Evaluate model performance with MSE

#### Quantile Model Training
1. Extract the same features as linear model
2. Prepare data for statsmodels
3. Define formula based on available features
4. Train models for different quantiles (0.25, 0.5, 0.75)
5. Evaluate with pseudo R-squared

### 2.2 Maker/Taker Model Training

Trains a binary classifier using logistic regression:

1. Extract features from orderbook:
   ```python
   # L2 book imbalance
   imbalance = (bid_volume - ask_volume) / (bid_volume + ask_volume)
   
   # Order size relative to best level
   rel_size_to_best = quantity / best_ask_qty
   
   # Order size relative to total liquidity
   rel_size_to_total = quantity / ask_volume
   ```

2. Build a pipeline with StandardScaler and LogisticRegression
3. Train on synthetic or historical data
4. Log feature importances to identify key predictors

## 3. Market Impact Calculation

### 3.1 Calibration Process

The Almgren-Chriss model auto-calibrates based on orderbook data:

```python
# Extract orderbook features
tightness = (near_bid_qty + near_ask_qty) / (total_bid_qty + total_ask_qty)

# Calibrate parameters
eta = 0.0002 * (1 + (1 - tightness))  # permanent impact factor
gamma = 1.5 * (0.5 + (1 - tightness))  # temporary impact exponent

# Adjust for volatility
temporary_impact_factor = 0.2 * (1 + volatility * 10)
permanent_impact_factor = 0.1 * (1 + volatility * 5)
```

### 3.2 Impact Calculation Process

1. Extract liquidity factor and depth from orderbook
2. Estimate market volume if not provided
3. Calculate order size as percentage of market volume
4. Compute temporary impact using power-law model
5. Compute permanent impact using linear model
6. Apply liquidity and volatility adjustments
7. Adjust direction based on buy/sell side
8. Calculate expected execution price

### 3.3 Optimal Execution Time

Based on the formula:
```
T* = sqrt((σ² × X) / (η × λ))
```

Where:
- σ = volatility
- X = order size relative to volume
- η = permanent impact factor
- λ = urgency/risk aversion factor

## 4. System Architecture and Optimizations

### 4.1 Overall Architecture

```
┌─────────────────┐           ┌───────────────┐
│  WebSocket      │◄──────────┤  Order Book   │
│  Client         │           │  Model        │
└─────────────────┘           └───────────────┘
        ▲                             ▲
        │                             │
        ▼                             ▼
┌─────────────────┐           ┌───────────────┐
│  UI Components  │◄──────────┤  Model Layer  │
│                 │           │  (Regression, │
└─────────────────┘           │   Impact)     │
        ▲                     └───────────────┘
        │                             ▲
        ▼                             │
┌─────────────────┐           ┌───────────────┐
│  Performance    │◄──────────┤  Fee          │
│  Tracking       │           │  Calculation  │
└─────────────────┘           └───────────────┘
```

### 4.2 Performance Optimizations

#### Latency Tracking
- Real-time measurement of market data to UI update latency
- Statistical analysis for averages, percentiles, and maximums
- Periodic logging every 10 seconds

```python
# Timing starts with market data
tick_id = market_data_tracker.start()

# Timing continues through UI updates
ui_event_id = ui_update_tracker.start()
self.root.after(0, lambda: self._update_ui_from_orderbook(tick_id, ui_event_id))
```

#### Memory Management
- Fixed-size deque for latency samples to prevent unbounded growth
- Cleanup of timing data for events that never complete
- Threading with proper locking for thread safety

#### Efficient Order Book Updates
- SortedDict for O(log n) insertions/deletions
- Incremental updates instead of full snapshots when possible
- Bid/ask caching for quick access to best prices

### 4.3 Communication Flow

1. **Market Data Reception**:
   - WebSocket client receives L2 orderbook data
   - Data is timestamped and processed
   - OrderBook model is updated

2. **UI Update Process**:
   - New orderbook triggers callback
   - UI is updated via tkinter event loop
   - Latency is measured and recorded

3. **Simulation Process**:
   - User inputs parameters
   - Models generate predictions
   - Results are displayed with cost breakdown

### 4.4 Real-time Performance Monitoring

The system includes a dedicated performance panel showing:
- UI update latency (average, max)
- Market data processing latency
- Update rate (updates/second)
- Sample count for statistical validity

This allows continuous monitoring and optimization of the system's performance during operation.

---

## Project Structure

- **models/** - Data models and business logic
  - `orderbook.py` - Order book implementation for handling market data
- **websocket/** - WebSocket clients for connecting to exchange APIs
- **ui/** - User interface components
  - `dashboard.py` - Tkinter-based dashboard for displaying market data
  - `simulator.py` - Trade simulator UI with input parameters and results
- **utils/** - Utility functions and helper classes

## Getting Started

### Prerequisites

- Python 3.7+
- Required packages (install using `pip install -r requirements.txt`)

### Installation

1. Clone the repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

### Running the WebSocket Client

To connect to the OKX BTC-USDT-SWAP order book and print the best bid and ask:

```
python websocket_client.py
```

### Running the Trade Simulator UI

To run the trade simulator with a graphical user interface:

```
python run_simulator.py
```

The simulator allows you to:
- Select trading pairs
- Choose order type (Market/Limit)
- Set quantity and other parameters
- View execution details, fees, and performance metrics

## Features

- Real-time order book data from crypto exchanges
- Order book visualization
- Best bid/ask tracking
- Spread calculation
- Trade simulation with fee calculation
- Market impact estimation

## License

MIT 
