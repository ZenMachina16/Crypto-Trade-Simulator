#!/usr/bin/env python3
"""Crypto trade simulator UI using Tkinter."""
import tkinter as tk
from tkinter import ttk
import logging
import numpy as np
import time
import threading
import asyncio
from models.slippage import SlippageEstimator, estimate_trade_cost
from models.orderbook import OrderBook
from models.impact import AlmgrenChriss, estimate_market_impact
from models.maker_taker import MakerTakerEstimator, generate_synthetic_training_data
from utils.fees import calculate_fee, estimate_fee_breakdown
from utils.performance import ui_update_tracker, market_data_tracker
from websocket.client import OrderBookClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('ui.simulator')

class SimulatorUI:
    """Main UI class for the crypto trade simulator."""
    
    def __init__(self, root):
        """Initialize the simulator UI.
        
        Args:
            root: tkinter root window
        """
        self.root = root
        root.title("Crypto Trade Simulator")
        root.geometry("1000x700")
        
        # Initialize models
        self.slippage_estimator = SlippageEstimator()
        self.maker_taker_estimator = MakerTakerEstimator()
        self.current_orderbook = None
        self.websocket_client = None
        self.update_count = 0
        
        # Start performance tracking
        ui_update_tracker.start_periodic_logging()
        
        # Create main frame with padding
        self.main_frame = ttk.Frame(root, padding="10")
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create panels
        self.create_panels()
        
        # Create status bar
        self.status_frame = ttk.Frame(root)
        self.status_frame.pack(fill=tk.X, side=tk.BOTTOM)
        
        # Status message
        self.status_var = tk.StringVar(value="Ready")
        self.status_bar = ttk.Label(self.status_frame, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(fill=tk.X, side=tk.LEFT, expand=True, pady=2)
        
        # Latency display
        self.latency_var = tk.StringVar(value="Latency: N/A")
        self.latency_label = ttk.Label(self.status_frame, textvariable=self.latency_var, relief=tk.SUNKEN, width=25)
        self.latency_label.pack(side=tk.RIGHT, pady=2, padx=(5, 0))
        
        # Update UI periodically
        self.start_ui_updater()
    
    def create_panels(self):
        """Create the input, output and monitoring panels."""
        # Create a frame for the top section (input and output)
        top_frame = ttk.Frame(self.main_frame)
        top_frame.pack(fill=tk.BOTH, expand=True, side=tk.TOP)
        
        # Left panel (input parameters)
        self.left_panel = ttk.LabelFrame(top_frame, text="Input Parameters", padding="10")
        self.left_panel.pack(fill=tk.BOTH, expand=True, side=tk.LEFT, padx=(0, 5))
        
        # Right panel (output display)
        self.right_panel = ttk.LabelFrame(top_frame, text="Simulation Results", padding="10")
        self.right_panel.pack(fill=tk.BOTH, expand=True, side=tk.RIGHT, padx=(5, 0))
        
        # Bottom panel (performance monitoring)
        self.monitor_panel = ttk.LabelFrame(self.main_frame, text="Performance Monitoring", padding="10")
        self.monitor_panel.pack(fill=tk.BOTH, side=tk.BOTTOM, pady=(10, 0))
        
        # Populate panels
        self.create_input_controls()
        self.create_output_display()
        self.create_monitor_display()
    
    def create_monitor_display(self):
        """Create performance monitoring display."""
        # Create a frame for the performance metrics
        perf_frame = ttk.Frame(self.monitor_panel)
        perf_frame.pack(fill=tk.BOTH, expand=True)
        
        # UI Update Latency
        ttk.Label(perf_frame, text="UI Update Latency:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.ui_latency_var = tk.StringVar(value="Avg: N/A, Max: N/A")
        ttk.Label(perf_frame, textvariable=self.ui_latency_var, width=30).grid(row=0, column=1, sticky=tk.W, pady=2)
        
        # Market Data Latency
        ttk.Label(perf_frame, text="Market Data Latency:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.md_latency_var = tk.StringVar(value="Avg: N/A, Max: N/A")
        ttk.Label(perf_frame, textvariable=self.md_latency_var, width=30).grid(row=1, column=1, sticky=tk.W, pady=2)
        
        # Transaction Rate
        ttk.Label(perf_frame, text="Update Rate:").grid(row=0, column=2, sticky=tk.W, pady=2, padx=(20, 0))
        self.rate_var = tk.StringVar(value="0 updates/sec")
        ttk.Label(perf_frame, textvariable=self.rate_var, width=20).grid(row=0, column=3, sticky=tk.W, pady=2)
        
        # Sample Count
        ttk.Label(perf_frame, text="Sample Count:").grid(row=1, column=2, sticky=tk.W, pady=2, padx=(20, 0))
        self.sample_var = tk.StringVar(value="UI: 0, MD: 0")
        ttk.Label(perf_frame, textvariable=self.sample_var, width=20).grid(row=1, column=3, sticky=tk.W, pady=2)
        
        # Live Connection Toggle
        ttk.Label(perf_frame, text="Live Connection:").grid(row=2, column=0, sticky=tk.W, pady=2)
        self.live_var = tk.BooleanVar(value=False)
        self.live_check = ttk.Checkbutton(perf_frame, variable=self.live_var, command=self.toggle_live_connection)
        self.live_check.grid(row=2, column=1, sticky=tk.W, pady=2)
        
        # Configure grid
        perf_frame.columnconfigure(1, weight=1)
        perf_frame.columnconfigure(3, weight=1)
    
    def create_input_controls(self):
        """Create input controls for simulation parameters."""
        # Asset dropdown
        ttk.Label(self.left_panel, text="Asset:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.asset_var = tk.StringVar(value="BTC-USDT-SWAP")
        asset_dropdown = ttk.Combobox(self.left_panel, textvariable=self.asset_var)
        asset_dropdown['values'] = ('BTC-USDT-SWAP', 'ETH-USDT-SWAP', 'SOL-USDT-SWAP', 'XRP-USDT-SWAP')
        asset_dropdown.grid(row=0, column=1, sticky=(tk.W, tk.E), pady=5)
        
        # Order type
        ttk.Label(self.left_panel, text="Order Type:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.order_type_var = tk.StringVar(value="Market")
        order_types = ttk.Frame(self.left_panel)
        order_types.grid(row=1, column=1, sticky=(tk.W, tk.E), pady=5)
        ttk.Radiobutton(order_types, text="Market", variable=self.order_type_var, value="Market").pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(order_types, text="Limit", variable=self.order_type_var, value="Limit").pack(side=tk.LEFT, padx=5)
        
        # Side (Buy/Sell)
        ttk.Label(self.left_panel, text="Side:").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.side_var = tk.StringVar(value="Buy")
        side_frame = ttk.Frame(self.left_panel)
        side_frame.grid(row=2, column=1, sticky=(tk.W, tk.E), pady=5)
        ttk.Radiobutton(side_frame, text="Buy", variable=self.side_var, value="Buy").pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(side_frame, text="Sell", variable=self.side_var, value="Sell").pack(side=tk.LEFT, padx=5)
        
        # Quantity
        ttk.Label(self.left_panel, text="Quantity:").grid(row=3, column=0, sticky=tk.W, pady=5)
        quantity_frame = ttk.Frame(self.left_panel)
        quantity_frame.grid(row=3, column=1, sticky=(tk.W, tk.E), pady=5)
        self.quantity_var = tk.StringVar(value="1.0")
        ttk.Entry(quantity_frame, textvariable=self.quantity_var, width=10).pack(side=tk.LEFT, padx=5)
        ttk.Label(quantity_frame, text="BTC").pack(side=tk.LEFT, padx=5)
        
        # Orderbook depth
        ttk.Label(self.left_panel, text="Orderbook Depth:").grid(row=4, column=0, sticky=tk.W, pady=5)
        depth_frame = ttk.Frame(self.left_panel)
        depth_frame.grid(row=4, column=1, sticky=(tk.W, tk.E), pady=5)
        self.depth_var = tk.StringVar(value="10")
        ttk.Entry(depth_frame, textvariable=self.depth_var, width=5).pack(side=tk.LEFT, padx=5)
        ttk.Label(depth_frame, text="levels").pack(side=tk.LEFT, padx=5)
        
        # Volatility
        ttk.Label(self.left_panel, text="Volatility (%):").grid(row=5, column=0, sticky=tk.W, pady=5)
        self.volatility_var = tk.StringVar(value="2.0")
        ttk.Entry(self.left_panel, textvariable=self.volatility_var, width=10).grid(row=5, column=1, sticky=tk.W, pady=5)
        
        # Impact model
        ttk.Label(self.left_panel, text="Impact Model:").grid(row=6, column=0, sticky=tk.W, pady=5)
        self.impact_model_var = tk.StringVar(value="Almgren-Chriss")
        impact_model_frame = ttk.Frame(self.left_panel)
        impact_model_frame.grid(row=6, column=1, sticky=(tk.W, tk.E), pady=5)
        ttk.Radiobutton(impact_model_frame, text="Simple", variable=self.impact_model_var, value="Simple").pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(impact_model_frame, text="Almgren-Chriss", variable=self.impact_model_var, value="Almgren-Chriss").pack(side=tk.LEFT, padx=5)
        
        # Fee tier
        ttk.Label(self.left_panel, text="Fee Tier:").grid(row=7, column=0, sticky=tk.W, pady=5)
        self.fee_tier_var = tk.StringVar(value="VIP 1")
        fee_tier_dropdown = ttk.Combobox(self.left_panel, textvariable=self.fee_tier_var)
        fee_tier_dropdown['values'] = ('Regular', 'VIP 1', 'VIP 2', 'VIP 3', 'VIP 4', 'VIP 5')
        fee_tier_dropdown.grid(row=7, column=1, sticky=(tk.W, tk.E), pady=5)
        
        # Execution speed
        ttk.Label(self.left_panel, text="Execution Speed:").grid(row=8, column=0, sticky=tk.W, pady=5)
        self.execution_speed_var = tk.StringVar(value="Normal")
        speed_dropdown = ttk.Combobox(self.left_panel, textvariable=self.execution_speed_var)
        speed_dropdown['values'] = ('Slow', 'Normal', 'Fast', 'Ultra')
        speed_dropdown.grid(row=8, column=1, sticky=(tk.W, tk.E), pady=5)
        
        # Regression model type
        ttk.Label(self.left_panel, text="Regression Model:").grid(row=9, column=0, sticky=tk.W, pady=5)
        self.model_var = tk.StringVar(value="Linear")
        model_frame = ttk.Frame(self.left_panel)
        model_frame.grid(row=9, column=1, sticky=(tk.W, tk.E), pady=5)
        ttk.Radiobutton(model_frame, text="Linear", variable=self.model_var, value="Linear").pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(model_frame, text="Quantile", variable=self.model_var, value="Quantile").pack(side=tk.LEFT, padx=5)
        
        # Separator
        ttk.Separator(self.left_panel, orient=tk.HORIZONTAL).grid(row=10, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10)
        
        # Simulate button
        self.simulate_button = ttk.Button(self.left_panel, text="Simulate Trade", command=self.simulate_trade)
        self.simulate_button.grid(row=11, column=0, columnspan=2, pady=10)
        
        # Reset button
        self.reset_button = ttk.Button(self.left_panel, text="Reset", command=self.reset_inputs)
        self.reset_button.grid(row=12, column=0, columnspan=2, pady=5)
        
        # Configure grid columns
        self.left_panel.columnconfigure(1, weight=1)
    
    def create_output_display(self):
        """Create output display for simulation results."""
        # Framework for output display
        output_frame = ttk.Frame(self.right_panel)
        output_frame.pack(fill=tk.BOTH, expand=True)
        
        # Style for result labels
        result_style = {'width': 15, 'anchor': tk.E}
        
        # Execution section
        ttk.Label(output_frame, text="Execution Details", font=("", 10, "bold")).grid(row=0, column=0, columnspan=2, sticky=tk.W, pady=(0, 5))
        
        # Entry price
        ttk.Label(output_frame, text="Entry Price:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.entry_price_var = tk.StringVar(value="N/A")
        ttk.Label(output_frame, textvariable=self.entry_price_var, **result_style).grid(row=1, column=1, sticky=tk.E, pady=2)
        
        # Market price
        ttk.Label(output_frame, text="Market Price:").grid(row=2, column=0, sticky=tk.W, pady=2)
        self.market_price_var = tk.StringVar(value="N/A")
        ttk.Label(output_frame, textvariable=self.market_price_var, **result_style).grid(row=2, column=1, sticky=tk.E, pady=2)
        
        # Slippage (with range)
        ttk.Label(output_frame, text="Slippage:").grid(row=3, column=0, sticky=tk.W, pady=2)
        self.slippage_var = tk.StringVar(value="N/A")
        ttk.Label(output_frame, textvariable=self.slippage_var, **result_style).grid(row=3, column=1, sticky=tk.E, pady=2)
        
        # Slippage range
        ttk.Label(output_frame, text="Slippage Range:").grid(row=4, column=0, sticky=tk.W, pady=2)
        self.slippage_range_var = tk.StringVar(value="N/A")
        ttk.Label(output_frame, textvariable=self.slippage_range_var, **result_style).grid(row=4, column=1, sticky=tk.E, pady=2)
        
        # Impact section
        ttk.Label(output_frame, text="Market Impact", font=("", 10, "bold")).grid(row=5, column=0, columnspan=2, sticky=tk.W, pady=(10, 5))
        
        # Temporary impact
        ttk.Label(output_frame, text="Temporary Impact:").grid(row=6, column=0, sticky=tk.W, pady=2)
        self.temp_impact_var = tk.StringVar(value="N/A")
        ttk.Label(output_frame, textvariable=self.temp_impact_var, **result_style).grid(row=6, column=1, sticky=tk.E, pady=2)
        
        # Permanent impact
        ttk.Label(output_frame, text="Permanent Impact:").grid(row=7, column=0, sticky=tk.W, pady=2)
        self.perm_impact_var = tk.StringVar(value="N/A")
        ttk.Label(output_frame, textvariable=self.perm_impact_var, **result_style).grid(row=7, column=1, sticky=tk.E, pady=2)
        
        # Total impact
        ttk.Label(output_frame, text="Total Impact:").grid(row=8, column=0, sticky=tk.W, pady=2)
        self.impact_var = tk.StringVar(value="N/A")
        ttk.Label(output_frame, textvariable=self.impact_var, **result_style).grid(row=8, column=1, sticky=tk.E, pady=2)
        
        # Order % of volume
        ttk.Label(output_frame, text="Order % of Volume:").grid(row=9, column=0, sticky=tk.W, pady=2)
        self.volume_pct_var = tk.StringVar(value="N/A")
        ttk.Label(output_frame, textvariable=self.volume_pct_var, **result_style).grid(row=9, column=1, sticky=tk.E, pady=2)
        
        # Optimal execution time
        ttk.Label(output_frame, text="Optimal Exec Time:").grid(row=10, column=0, sticky=tk.W, pady=2)
        self.execution_time_var = tk.StringVar(value="N/A")
        ttk.Label(output_frame, textvariable=self.execution_time_var, **result_style).grid(row=10, column=1, sticky=tk.E, pady=2)
        
        # Separator
        ttk.Separator(output_frame, orient=tk.HORIZONTAL).grid(row=11, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10)
        
        # Fees section
        ttk.Label(output_frame, text="Fees & Costs", font=("", 10, "bold")).grid(row=12, column=0, columnspan=2, sticky=tk.W, pady=(0, 5))
        
        # Trading fee
        ttk.Label(output_frame, text="Trading Fee:").grid(row=13, column=0, sticky=tk.W, pady=2)
        self.fee_var = tk.StringVar(value="N/A")
        ttk.Label(output_frame, textvariable=self.fee_var, **result_style).grid(row=13, column=1, sticky=tk.E, pady=2)
        
        # Maker/Taker
        ttk.Label(output_frame, text="Maker/Taker:").grid(row=14, column=0, sticky=tk.W, pady=2)
        self.maker_taker_var = tk.StringVar(value="N/A")
        ttk.Label(output_frame, textvariable=self.maker_taker_var, **result_style).grid(row=14, column=1, sticky=tk.E, pady=2)
        
        # Maker fee
        ttk.Label(output_frame, text="Maker Fee:").grid(row=15, column=0, sticky=tk.W, pady=2)
        self.maker_fee_var = tk.StringVar(value="N/A")
        ttk.Label(output_frame, textvariable=self.maker_fee_var, **result_style).grid(row=15, column=1, sticky=tk.E, pady=2)
        
        # Taker fee
        ttk.Label(output_frame, text="Taker Fee:").grid(row=16, column=0, sticky=tk.W, pady=2)
        self.taker_fee_var = tk.StringVar(value="N/A")
        ttk.Label(output_frame, textvariable=self.taker_fee_var, **result_style).grid(row=16, column=1, sticky=tk.E, pady=2)
        
        # Net Cost
        ttk.Label(output_frame, text="Net Cost:").grid(row=17, column=0, sticky=tk.W, pady=2)
        self.net_cost_var = tk.StringVar(value="N/A")
        ttk.Label(output_frame, textvariable=self.net_cost_var, **result_style).grid(row=17, column=1, sticky=tk.E, pady=2)
        
        # Cost Range
        ttk.Label(output_frame, text="Cost Range:").grid(row=18, column=0, sticky=tk.W, pady=2)
        self.cost_range_var = tk.StringVar(value="N/A")
        ttk.Label(output_frame, textvariable=self.cost_range_var, **result_style).grid(row=18, column=1, sticky=tk.E, pady=2)
        
        # Separator
        ttk.Separator(output_frame, orient=tk.HORIZONTAL).grid(row=19, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10)
        
        # Performance section
        ttk.Label(output_frame, text="Performance Metrics", font=("", 10, "bold")).grid(row=20, column=0, columnspan=2, sticky=tk.W, pady=(0, 5))
        
        # Latency
        ttk.Label(output_frame, text="Latency:").grid(row=21, column=0, sticky=tk.W, pady=2)
        self.latency_var = tk.StringVar(value="N/A")
        ttk.Label(output_frame, textvariable=self.latency_var, **result_style).grid(row=21, column=1, sticky=tk.E, pady=2)
        
        # Fill rate
        ttk.Label(output_frame, text="Fill Rate:").grid(row=22, column=0, sticky=tk.W, pady=2)
        self.fill_rate_var = tk.StringVar(value="N/A")
        ttk.Label(output_frame, textvariable=self.fill_rate_var, **result_style).grid(row=22, column=1, sticky=tk.E, pady=2)
        
        # Model information
        ttk.Label(output_frame, text="Model Type:").grid(row=23, column=0, sticky=tk.W, pady=2)
        self.model_type_var = tk.StringVar(value="N/A")
        ttk.Label(output_frame, textvariable=self.model_type_var, **result_style).grid(row=23, column=1, sticky=tk.E, pady=2)
        
        # Configure grid columns
        output_frame.columnconfigure(1, weight=1)
    
    def _create_synthetic_orderbook(self, asset, depth=10, volatility=2.0):
        """Create a synthetic orderbook for simulation purposes.
        
        Args:
            asset (str): Trading pair symbol
            depth (int): Number of levels to generate
            volatility (float): Volatility percentage to simulate
            
        Returns:
            OrderBook: Generated orderbook
        """
        # Base prices for different assets
        base_prices = {
            'BTC-USDT-SWAP': 65000.0,
            'ETH-USDT-SWAP': 3500.0,
            'SOL-USDT-SWAP': 140.0,
            'XRP-USDT-SWAP': 0.55,
        }
        
        base_price = base_prices.get(asset, 65000.0)
        
        # Create synthetic order book
        orderbook = OrderBook(asset)
        
        # Generate random bids and asks with appropriate depth
        spread = base_price * 0.0005  # 0.05% spread
        
        # Create bids (buy orders), descending from mid - spread/2
        bid_start = base_price - spread/2
        
        bids = []
        for i in range(depth):
            price = bid_start * (1 - 0.0001 * i - np.random.random() * 0.0001 * volatility)
            quantity = 1.0 + np.random.exponential(3.0) * (1 + np.random.random() * volatility)
            bids.append((price, quantity))
        
        # Create asks (sell orders), ascending from mid + spread/2
        ask_start = base_price + spread/2
        
        asks = []
        for i in range(depth):
            price = ask_start * (1 + 0.0001 * i + np.random.random() * 0.0001 * volatility)
            quantity = 1.0 + np.random.exponential(2.0) * (1 + np.random.random() * volatility)
            asks.append((price, quantity))
        
        # Update orderbook with this data
        snapshot_data = {
            'bids': bids,
            'asks': asks,
            'timestamp': 1000000000,
            'action': 'snapshot'
        }
        
        orderbook.update(snapshot_data)
        return orderbook
    
    def simulate_trade(self):
        """Simulate a trade based on current input parameters and update the display."""
        # Get input values
        asset = self.asset_var.get()
        order_type = self.order_type_var.get()
        side = self.side_var.get()
        model_type = self.model_var.get()
        impact_model = self.impact_model_var.get()
        
        try:
            quantity = float(self.quantity_var.get())
            volatility = float(self.volatility_var.get()) / 100.0  # Convert percentage to decimal
            depth = int(self.depth_var.get())
        except ValueError:
            self.status_var.set("Error: Invalid input values")
            return
            
        fee_tier = self.fee_tier_var.get()
        execution_speed = self.execution_speed_var.get()
        
        # Log the inputs
        logger.info(f"Simulating {side} {order_type} order for {quantity} {asset} with {fee_tier} tier")
        
        # Create synthetic orderbook if not using real data
        self.current_orderbook = self._create_synthetic_orderbook(asset, depth, volatility * 100)
        
        # Generate some synthetic historical trades for training the regression models
        mock_trades = self._generate_training_data(asset, 50, volatility * 100)
        mock_orderbooks = [self._create_synthetic_orderbook(asset, depth, volatility * 100) for _ in range(50)]
        
        # Train the slippage model
        if model_type == "Linear":
            self.slippage_estimator.train_linear_model(mock_trades, mock_orderbooks)
            self.model_type_var.set("Linear Regression")
        else:
            self.slippage_estimator.train_quantile_model(mock_trades, mock_orderbooks)
            self.model_type_var.set("Quantile Regression")
        
        # Train the maker/taker model
        obs, qtys, labels = generate_synthetic_training_data(self.current_orderbook, 100)
        self.maker_taker_estimator.train(obs, qtys, labels)
        
        # Get market price
        if side.lower() == "buy":
            market_price, _ = self.current_orderbook.best_ask() or (None, None)
        else:
            market_price, _ = self.current_orderbook.best_bid() or (None, None)
        
        if not market_price:
            self.status_var.set("Error: Could not get market price")
            return
        
        # Estimate slippage using the trained model
        cost_estimates = estimate_trade_cost(quantity, self.current_orderbook, side.lower(), self.slippage_estimator)
        slippage_pct = cost_estimates['slippage_pct']
        slippage_range = cost_estimates['slippage_range']
        
        # Estimate market impact using Almgren-Chriss model
        if impact_model == "Almgren-Chriss":
            # Use Almgren-Chriss model
            impact_result = estimate_market_impact(market_price, quantity, self.current_orderbook, side.lower(), volatility)
            
            # Extract impact metrics
            temporary_impact_pct = impact_result['temporary_impact_pct']
            permanent_impact_pct = impact_result['permanent_impact_pct']
            total_impact_pct = impact_result['total_impact_pct']
            volume_pct = impact_result['order_pct_of_volume']
            
            # Calculate optimal execution time
            ac_model = AlmgrenChriss.calibrate_from_orderbook(self.current_orderbook, volatility)
            execution_time = ac_model.calculate_optimal_execution_time(quantity, market_price)
            
            # Use the model's expected execution price
            entry_price = impact_result['expected_execution_price']
        else:
            # Use simple impact model (impact proportional to order size and volatility)
            # These are simplified estimates for comparison
            volume_factor = min(quantity / 10.0, 1.0)  # Cap at 100%
            temporary_impact_pct = slippage_pct * 0.8  # 80% of slippage is temporary
            permanent_impact_pct = slippage_pct * 0.2  # 20% of slippage is permanent
            total_impact_pct = slippage_pct
            
            # Calculate order percentage of volume (dummy value)
            volume_pct = volume_factor * 10  # 0-10%
            
            # No optimal execution time for simple model
            execution_time = 0.0
            
            # Use the slippage-based execution price
            entry_price = cost_estimates['estimated_execution_price']
        
        # Calculate fees using the fee calculator with orderbook data for maker/taker estimation
        fee_result = calculate_fee(entry_price, quantity, fee_tier, order_type, self.current_orderbook)
        fee_amount = fee_result['fee_amount']
        fee_rate = fee_result['fee_rate']
        maker_taker_ratio = fee_result['maker_taker_ratio']
        
        # Get detailed fee breakdown
        fee_breakdown = estimate_fee_breakdown(entry_price * quantity, fee_tier, order_type, 
                                              self.current_orderbook, quantity)
        maker_fee = fee_breakdown['maker_fee']
        taker_fee = fee_breakdown['taker_fee']
        
        # Calculate net cost
        if side.lower() == 'buy':
            net_cost = (entry_price * quantity) + fee_amount
        else:
            net_cost = (entry_price * quantity) - fee_amount
        
        # Calculate cost range using slippage range
        cost_range = cost_estimates['total_cost_range']
        
        # Latency based on execution speed
        latencies = {
            "Slow": "150-200ms",
            "Normal": "80-120ms",
            "Fast": "40-60ms",
            "Ultra": "10-20ms"
        }
        latency = latencies.get(execution_speed, "80-120ms")
        
        # Fill rate
        fill_rate = "100%" if order_type == "Market" else "95%"
        
        # Update output display
        self.entry_price_var.set(f"${entry_price:.2f}")
        self.market_price_var.set(f"${market_price:.2f}")
        self.slippage_var.set(f"{slippage_pct:.4f}%")
        self.slippage_range_var.set(f"{slippage_range[0]:.4f}% - {slippage_range[1]:.4f}%")
        
        # Update impact metrics
        self.temp_impact_var.set(f"{temporary_impact_pct:.4f}%")
        self.perm_impact_var.set(f"{permanent_impact_pct:.4f}%")
        self.impact_var.set(f"{total_impact_pct:.4f}%")
        self.volume_pct_var.set(f"{volume_pct:.2f}%")
        
        # Show optimal execution time if available
        if execution_time > 0:
            if execution_time < 1:
                self.execution_time_var.set(f"{execution_time*60:.1f} minutes")
            else:
                self.execution_time_var.set(f"{execution_time:.2f} hours")
        else:
            self.execution_time_var.set("N/A")
        
        # Update fee information
        self.fee_var.set(f"${fee_amount:.2f} ({fee_rate*100:.4f}%)")
        self.maker_taker_var.set(maker_taker_ratio)
        self.maker_fee_var.set(f"${maker_fee:.2f}")
        self.taker_fee_var.set(f"${taker_fee:.2f}")
        self.net_cost_var.set(f"${net_cost:.2f}")
        self.cost_range_var.set(f"${cost_range[0]:.2f} - ${cost_range[1]:.2f}")
        
        # Update performance metrics
        self.latency_var.set(latency)
        self.fill_rate_var.set(fill_rate)
        
        # Update status
        self.status_var.set(f"Trade simulation completed for {quantity} {asset}")
    
    def _generate_training_data(self, asset, num_trades=50, volatility=2.0):
        """Generate synthetic trade data for model training.
        
        Args:
            asset (str): Trading pair symbol
            num_trades (int): Number of trades to generate
            volatility (float): Volatility factor
            
        Returns:
            list: List of synthetic trade data
        """
        # Base prices for different assets
        base_prices = {
            'BTC-USDT-SWAP': 65000.0,
            'ETH-USDT-SWAP': 3500.0,
            'SOL-USDT-SWAP': 140.0,
            'XRP-USDT-SWAP': 0.55,
        }
        
        base_price = base_prices.get(asset, 65000.0)
        trades = []
        
        for _ in range(num_trades):
            # Randomize market price around base
            market_price = base_price * (1 + np.random.normal(0, 0.001))
            
            # Random quantity between 0.1 and 20
            quantity = 0.1 + np.random.exponential(2.0)
            
            # Random side
            side = 'buy' if np.random.random() > 0.5 else 'sell'
            
            # Generate reasonable slippage based on quantity and volatility
            # Higher quantity and volatility should lead to higher slippage
            base_slippage = 0.01 + (quantity / 20) * 0.1 + (volatility / 10) * 0.05
            
            # Add some noise to slippage
            slippage = max(0, base_slippage + np.random.normal(0, base_slippage/3))
            
            # Calculate executed price with slippage
            if side == 'buy':
                executed_price = market_price * (1 + slippage)
            else:
                executed_price = market_price * (1 - slippage)
            
            trades.append({
                'side': side,
                'quantity': quantity,
                'market_price': market_price,
                'executed_price': executed_price
            })
            
        return trades
    
    def reset_inputs(self):
        """Reset all input fields to default values."""
        self.asset_var.set("BTC-USDT-SWAP")
        self.order_type_var.set("Market")
        self.side_var.set("Buy")
        self.quantity_var.set("1.0")
        self.volatility_var.set("2.0")
        self.depth_var.set("10")
        self.fee_tier_var.set("VIP 1")
        self.execution_speed_var.set("Normal")
        self.model_var.set("Linear")
        self.impact_model_var.set("Almgren-Chriss")
        
        # Reset outputs
        for var in [self.entry_price_var, self.market_price_var, self.slippage_var, 
                   self.slippage_range_var, self.temp_impact_var, self.perm_impact_var, 
                   self.impact_var, self.volume_pct_var, self.execution_time_var,
                   self.fee_var, self.maker_taker_var, self.maker_fee_var, self.taker_fee_var,
                   self.net_cost_var, self.cost_range_var,
                   self.latency_var, self.fill_rate_var, self.model_type_var]:
            var.set("N/A")
        
        self.status_var.set("Ready")
    
    def handle_orderbook_update(self, orderbook, tick_id):
        """Handle an update from the WebSocket client.
        
        Args:
            orderbook (OrderBook): Updated orderbook
            tick_id: Event ID for latency tracking
        """
        # Start UI update timing
        ui_event_id = ui_update_tracker.start()
        
        # Update the current orderbook
        self.current_orderbook = orderbook
        
        # Schedule UI update on the main thread
        self.root.after(0, lambda: self._update_ui_from_orderbook(tick_id, ui_event_id))
        
    def _update_ui_from_orderbook(self, tick_id, ui_event_id):
        """Update UI based on the latest orderbook data.
        
        Args:
            tick_id: Market data event ID for timing
            ui_event_id: UI update event ID for timing
        """
        if not self.current_orderbook:
            return
            
        # Get current orderbook data
        bid_price, bid_qty = self.current_orderbook.best_bid() or (None, None)
        ask_price, ask_qty = self.current_orderbook.best_ask() or (None, None)
        
        if bid_price is not None and ask_price is not None:
            # Update market price in the UI
            market_price = (bid_price + ask_price) / 2
            self.market_price_var.set(f"${market_price:.2f}")
            
            # Update spread information
            spread = ask_price - bid_price
            spread_pct = (spread / bid_price) * 100
            
            # Update status with orderbook information
            self.status_var.set(
                f"Live OrderBook: Bid: {bid_price:.2f} @ {bid_qty:.6f}, "
                f"Ask: {ask_price:.2f} @ {ask_qty:.6f}, "
                f"Spread: {spread:.2f} ({spread_pct:.4f}%)"
            )
            
            # Increment update counter
            self.update_count += 1
        
        # Stop market data timing
        latency_ms = market_data_tracker.stop(tick_id)
        if latency_ms:
            self.latency_var.set(f"Latency: {latency_ms:.2f} ms")
            
        # Stop UI update timing
        ui_update_tracker.stop(ui_event_id)
    
    def toggle_live_connection(self):
        """Toggle live market data connection."""
        if self.live_var.get():
            # Connect to live data
            self._start_live_connection()
        else:
            # Disconnect from live data
            self._stop_live_connection()
    
    def _start_live_connection(self):
        """Start live connection to market data feed."""
        if self.websocket_client:
            return
            
        try:
            # Get symbol from UI
            symbol = self.asset_var.get()
            
            # Create and start websocket client in a separate thread
            uri = f"wss://ws.gomarket-cpp.goquant.io/ws/l2-orderbook/okx/{symbol}"
            self.websocket_client = OrderBookClient(uri, symbol, callback=self.handle_orderbook_update)
            
            # Start the client in a background thread
            def run_client():
                asyncio.run(self.websocket_client.connect())
                
            self.ws_thread = threading.Thread(target=run_client, daemon=True)
            self.ws_thread.start()
            
            logger.info(f"Started live connection to {symbol}")
            self.status_var.set(f"Connected to live {symbol} feed")
            
        except Exception as e:
            logger.error(f"Error starting live connection: {e}", exc_info=True)
            self.status_var.set(f"Error connecting: {str(e)}")
            self.live_var.set(False)
    
    def _stop_live_connection(self):
        """Stop live market data connection."""
        # The WebSocket client will be closed when the thread terminates
        self.websocket_client = None
        self.status_var.set("Disconnected from live feed")
        logger.info("Stopped live connection")
    
    def start_ui_updater(self):
        """Start a timer to update performance statistics periodically."""
        self._update_performance_stats()
        # Schedule next update in 1 second
        self.root.after(1000, self.start_ui_updater)
    
    def _update_performance_stats(self):
        """Update performance statistics display."""
        # Get UI latency stats
        ui_stats = ui_update_tracker.get_statistics()
        self.ui_latency_var.set(
            f"Avg: {ui_stats['avg']:.2f} ms, Max: {ui_stats['max']:.2f} ms"
        )
        
        # Get market data latency stats
        md_stats = market_data_tracker.get_statistics()
        self.md_latency_var.set(
            f"Avg: {md_stats['avg']:.2f} ms, Max: {md_stats['max']:.2f} ms"
        )
        
        # Calculate update rate
        current_time = time.time()
        elapsed = current_time - getattr(self, 'last_rate_time', current_time - 1)
        updates = self.update_count - getattr(self, 'last_update_count', 0)
        
        if elapsed > 0:
            rate = updates / elapsed
            self.rate_var.set(f"{rate:.1f} updates/sec")
        
        # Update sample counts
        self.sample_var.set(f"UI: {ui_stats['samples']}, MD: {md_stats['samples']}")
        
        # Save current values for next calculation
        self.last_rate_time = current_time
        self.last_update_count = self.update_count


def create_simulator_ui():
    """Create and return a simulator UI instance."""
    root = tk.Tk()
    
    # Set theme (if available)
    try:
        style = ttk.Style()
        style.theme_use('clam')  # Try to use a more modern theme
    except tk.TclError:
        pass  # Use default theme if 'clam' is not available
    
    simulator = SimulatorUI(root)
    return root, simulator


if __name__ == "__main__":
    root, simulator = create_simulator_ui()
    root.mainloop() 