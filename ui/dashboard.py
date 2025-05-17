"""Dashboard for displaying orderbook and trade data."""
import asyncio
import tkinter as tk
from tkinter import ttk

class Dashboard:
    def __init__(self, root):
        """Initialize dashboard UI.
        
        Args:
            root: tkinter root window
        """
        self.root = root
        root.title("Crypto Trade Simulator")
        root.geometry("800x600")
        
        # Create main frame
        self.main_frame = ttk.Frame(root, padding="10")
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # OrderBook display
        self.orderbook_frame = ttk.LabelFrame(self.main_frame, text="Order Book", padding="10")
        self.orderbook_frame.pack(fill=tk.BOTH, expand=True, side=tk.LEFT)
        
        # Asks Frame (top part of orderbook)
        self.asks_frame = ttk.Frame(self.orderbook_frame)
        self.asks_frame.pack(fill=tk.BOTH, expand=True)
        
        # Separator
        ttk.Separator(self.orderbook_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=5)
        
        # Bids Frame (bottom part of orderbook)
        self.bids_frame = ttk.Frame(self.orderbook_frame)
        self.bids_frame.pack(fill=tk.BOTH, expand=True)
        
        # Asks and Bids Headers
        self.create_headers()
        
        # Trading Panel
        self.trading_frame = ttk.LabelFrame(self.main_frame, text="Trading", padding="10")
        self.trading_frame.pack(fill=tk.BOTH, expand=True, side=tk.RIGHT)
        
        # Status Bar
        self.status_var = tk.StringVar(value="Disconnected")
        self.status_bar = ttk.Label(root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(fill=tk.X, side=tk.BOTTOM, pady=2)
    
    def create_headers(self):
        """Create headers for the asks and bids tables."""
        # Asks headers
        asks_header = ttk.Frame(self.asks_frame)
        asks_header.pack(fill=tk.X)
        ttk.Label(asks_header, text="Price", width=10).pack(side=tk.LEFT)
        ttk.Label(asks_header, text="Quantity", width=10).pack(side=tk.LEFT)
        ttk.Label(asks_header, text="Total", width=10).pack(side=tk.LEFT)
        
        # Bids headers
        bids_header = ttk.Frame(self.bids_frame)
        bids_header.pack(fill=tk.X)
        ttk.Label(bids_header, text="Price", width=10).pack(side=tk.LEFT)
        ttk.Label(bids_header, text="Quantity", width=10).pack(side=tk.LEFT)
        ttk.Label(bids_header, text="Total", width=10).pack(side=tk.LEFT)
    
    def update_orderbook(self, orderbook):
        """Update the orderbook display.
        
        Args:
            orderbook: OrderBook instance with market data
        """
        # Clear current orderbook display
        for widget in self.asks_frame.winfo_children()[1:]:
            widget.destroy()
        for widget in self.bids_frame.winfo_children()[1:]:
            widget.destroy()
        
        # Update asks (sorted by price ascending)
        for i, (price, qty) in enumerate(sorted(orderbook.asks.items())[:10]):
            if i >= 10:  # Show only top 10 levels
                break
            row = ttk.Frame(self.asks_frame)
            row.pack(fill=tk.X)
            ttk.Label(row, text=f"{price:.2f}", foreground="red", width=10).pack(side=tk.LEFT)
            ttk.Label(row, text=f"{qty:.6f}", width=10).pack(side=tk.LEFT)
            ttk.Label(row, text=f"{price * qty:.2f}", width=10).pack(side=tk.LEFT)
        
        # Update bids (sorted by price descending)
        for i, (price, qty) in enumerate(sorted(orderbook.bids.items(), reverse=True)[:10]):
            if i >= 10:  # Show only top 10 levels
                break
            row = ttk.Frame(self.bids_frame)
            row.pack(fill=tk.X)
            ttk.Label(row, text=f"{price:.2f}", foreground="green", width=10).pack(side=tk.LEFT)
            ttk.Label(row, text=f"{qty:.6f}", width=10).pack(side=tk.LEFT)
            ttk.Label(row, text=f"{price * qty:.2f}", width=10).pack(side=tk.LEFT)
    
    def set_status(self, status):
        """Update status bar text.
        
        Args:
            status (str): Status message to display
        """
        self.status_var.set(status)
    
    def run(self):
        """Start the UI event loop."""
        self.root.mainloop()


async def update_ui_periodically(dashboard, orderbook):
    """Update the UI at regular intervals.
    
    Args:
        dashboard: Dashboard instance to update
        orderbook: OrderBook instance with market data
    """
    while True:
        # Schedule UI update on the main thread
        dashboard.root.after(0, dashboard.update_orderbook, orderbook)
        await asyncio.sleep(0.1)  # Update 10 times per second


def create_dashboard():
    """Create and return a dashboard instance."""
    root = tk.Tk()
    dashboard = Dashboard(root)
    return dashboard 