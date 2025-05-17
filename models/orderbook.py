"""Order book model for handling L2 market data."""
from sortedcontainers import SortedDict
import time

class OrderBook:
    def __init__(self, symbol):
        """Initialize an empty order book for a given symbol.
        
        Args:
            symbol (str): Trading pair symbol (e.g., 'BTC-USDT-SWAP')
        """
        self.symbol = symbol
        # SortedDict with reverse=True for bids (highest price first)
        self.bids = SortedDict(lambda k: -float(k))  # Price -> Quantity, descending sort
        # SortedDict with default sorting for asks (lowest price first)
        self.asks = SortedDict()  # Price -> Quantity, ascending sort
        self.timestamp = None
        self.last_update_time = None
        self.sequence = None
    
    def handle_snapshot(self, data):
        """Process a full snapshot update of the order book.
        
        Args:
            data (dict): Order book snapshot data from WebSocket
        """
        self.timestamp = data.get('timestamp')
        self.sequence = data.get('sequence')
        self.last_update_time = time.time()
        
        # Clear existing book
        self.bids.clear()
        self.asks.clear()
        
        # Process all bids
        if 'bids' in data and isinstance(data['bids'], list):
            for price_qty in data['bids']:
                if len(price_qty) >= 2:
                    price, qty = price_qty[0], price_qty[1]
                    if float(qty) > 0:
                        self.bids[float(price)] = float(qty)
        
        # Process all asks
        if 'asks' in data and isinstance(data['asks'], list):
            for price_qty in data['asks']:
                if len(price_qty) >= 2:
                    price, qty = price_qty[0], price_qty[1]
                    if float(qty) > 0:
                        self.asks[float(price)] = float(qty)
    
    def handle_delta(self, data):
        """Process an incremental update to the order book.
        
        Args:
            data (dict): Order book delta update from WebSocket
        """
        self.timestamp = data.get('timestamp')
        self.sequence = data.get('sequence')
        self.last_update_time = time.time()
        
        # Process bid updates
        if 'bids' in data and isinstance(data['bids'], list):
            for price_qty in data['bids']:
                if len(price_qty) >= 2:
                    price, qty = float(price_qty[0]), float(price_qty[1])
                    if qty == 0:
                        # Remove price level
                        self.bids.pop(price, None)
                    else:
                        # Add or update price level
                        self.bids[price] = qty
        
        # Process ask updates
        if 'asks' in data and isinstance(data['asks'], list):
            for price_qty in data['asks']:
                if len(price_qty) >= 2:
                    price, qty = float(price_qty[0]), float(price_qty[1])
                    if qty == 0:
                        # Remove price level
                        self.asks.pop(price, None)
                    else:
                        # Add or update price level
                        self.asks[price] = qty
    
    def update(self, data):
        """Update order book with new data.
        
        Args:
            data (dict): Order book data from WebSocket
        """
        if data.get('action') == 'snapshot':
            self.handle_snapshot(data)
        else:
            # Default to delta update
            self.handle_delta(data)
    
    def best_bid(self):
        """Get the highest bid price and quantity."""
        if not self.bids:
            return None, None
        try:
            # Get the first item (highest bid) from sorted bids
            price = next(iter(self.bids))
            return price, self.bids[price]
        except (StopIteration, KeyError):
            return None, None
    
    def best_ask(self):
        """Get the lowest ask price and quantity."""
        if not self.asks:
            return None, None
        try:
            # Get the first item (lowest ask) from sorted asks
            price = next(iter(self.asks))
            return price, self.asks[price]
        except (StopIteration, KeyError):
            return None, None
    
    def mid_price(self):
        """Calculate the mid price between best bid and best ask."""
        bid_price, _ = self.best_bid()
        ask_price, _ = self.best_ask()
        
        if bid_price is None or ask_price is None:
            return None
        
        return (bid_price + ask_price) / 2
    
    def spread(self):
        """Calculate the spread between best bid and ask prices."""
        bid_price, _ = self.best_bid()
        ask_price, _ = self.best_ask()
        
        if bid_price is None or ask_price is None:
            return None
        
        return ask_price - bid_price
    
    def spread_percentage(self):
        """Calculate the spread as a percentage of the best bid."""
        spread = self.spread()
        bid_price, _ = self.best_bid()
        
        if spread is None or bid_price is None or bid_price == 0:
            return None
        
        return (spread / bid_price) * 100
    
    def get_price_levels(self, side, depth=10):
        """Get a specific number of price levels from the order book.
        
        Args:
            side (str): 'bids' or 'asks'
            depth (int): Number of price levels to return
            
        Returns:
            list: List of (price, quantity) tuples
        """
        if side.lower() == 'bids':
            book = self.bids
        elif side.lower() == 'asks':
            book = self.asks
        else:
            raise ValueError("Side must be 'bids' or 'asks'")
        
        levels = []
        for price in list(book.keys())[:depth]:
            levels.append((price, book[price]))
        
        return levels
    
    def __str__(self):
        best_bid_price, best_bid_qty = self.best_bid() or (None, None)
        best_ask_price, best_ask_qty = self.best_ask() or (None, None)
        
        bid_str = f"{best_bid_price:.2f} @ {best_bid_qty:.6f}" if best_bid_price is not None else "None"
        ask_str = f"{best_ask_price:.2f} @ {best_ask_qty:.6f}" if best_ask_price is not None else "None"
        spread = self.spread()
        spread_str = f"{spread:.2f}" if spread is not None else "N/A"
        
        return (f"OrderBook({self.symbol}): "
                f"Best Bid: {bid_str}, "
                f"Best Ask: {ask_str}, "
                f"Spread: {spread_str}") 