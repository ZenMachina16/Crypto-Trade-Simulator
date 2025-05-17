# Crypto Trade Simulator

A real-time cryptocurrency trading simulator that connects to market data feeds and provides a simple interface for simulating trades.

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