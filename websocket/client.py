#!/usr/bin/env python3
"""WebSocket client for connecting to cryptocurrency exchange order book streams."""
import asyncio
import json
import logging
import time
import websockets
from models.orderbook import OrderBook
from utils.performance import market_data_tracker

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('websocket.client')

class OrderBookClient:
    """Client for connecting to order book WebSocket feeds and maintaining order book state."""
    
    def __init__(self, uri, symbol, callback=None):
        """Initialize the order book client.
        
        Args:
            uri (str): WebSocket URI to connect to
            symbol (str): Trading pair symbol
            callback (callable): Optional callback for order book updates
        """
        self.uri = uri
        self.symbol = symbol
        self.orderbook = OrderBook(symbol)
        self.callback = callback
        self.connected = False
        self.reconnect_delay = 5  # Initial reconnect delay in seconds
        self.max_reconnect_delay = 60  # Maximum reconnect delay in seconds
        self.start_time = time.time()
        
        # Start tracking market data latency
        market_data_tracker.start_periodic_logging()
    
    async def connect(self):
        """Establish connection to the WebSocket server."""
        logger.info(f"Connecting to {self.uri}...")
        
        while True:
            try:
                async with websockets.connect(self.uri) as websocket:
                    logger.info(f"Connection established for {self.symbol}")
                    self.connected = True
                    self.reconnect_delay = 5  # Reset reconnect delay
                    
                    await self._message_handler(websocket)
            
            except websockets.exceptions.ConnectionClosed as e:
                logger.warning(f"Connection closed: {e}")
                self.connected = False
            
            except Exception as e:
                logger.error(f"Error: {e}", exc_info=True)
                self.connected = False
            
            # Implement exponential backoff for reconnection
            logger.info(f"Reconnecting in {self.reconnect_delay} seconds...")
            await asyncio.sleep(self.reconnect_delay)
            self.reconnect_delay = min(self.reconnect_delay * 1.5, self.max_reconnect_delay)
    
    async def _message_handler(self, websocket):
        """Handle incoming WebSocket messages.
        
        Args:
            websocket: WebSocket connection
        """
        while True:
            try:
                # Start timing market data receipt
                tick_id = market_data_tracker.start()
                
                # Receive message from the WebSocket
                message = await websocket.recv()
                
                # Parse JSON payload to Python dict
                data = json.loads(message)
                
                # Add receipt timestamp
                data['received_at'] = time.time()
                data['client_latency'] = 0  # Will be updated by the callback
                
                # Update the orderbook with the new data
                self.orderbook.update(data)
                
                # Call the callback with the updated orderbook
                if self.callback:
                    self.callback(self.orderbook, tick_id)
                else:
                    # Stop timing if no callback is registered
                    market_data_tracker.stop(tick_id)
                
                # Print orderbook information
                await self._print_orderbook_info()
                
            except websockets.exceptions.ConnectionClosed:
                logger.warning("Connection closed")
                break
                
            except json.JSONDecodeError as e:
                logger.error(f"JSON decode error: {e}, message: {message[:100]}...")
                
            except Exception as e:
                logger.error(f"Error handling message: {e}", exc_info=True)
    
    async def _print_orderbook_info(self):
        """Print information about the current state of the order book."""
        bid_price, bid_qty = self.orderbook.best_bid() or (None, None)
        ask_price, ask_qty = self.orderbook.best_ask() or (None, None)
        
        if bid_price is not None and ask_price is not None:
            spread = ask_price - bid_price
            spread_pct = (spread / bid_price) * 100
            
            logger.info(
                f"L2 OrderBook: Bid: {bid_price:.2f} @ {bid_qty:.6f} | "
                f"Ask: {ask_price:.2f} @ {ask_qty:.6f} | "
                f"Spread: {spread:.2f} ({spread_pct:.4f}%) | "
                f"Levels: {len(self.orderbook.bids)}b/{len(self.orderbook.asks)}a"
            )

async def main():
    """Main entry point for the WebSocket client."""
    # L2 Order Book WebSocket URL
    uri = "wss://ws.gomarket-cpp.goquant.io/ws/l2-orderbook/okx/BTC-USDT-SWAP"
    symbol = "BTC-USDT-SWAP"
    
    client = OrderBookClient(uri, symbol)
    await client.connect()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Program interrupted by user. Exiting...")
    except Exception as e:
        logger.error(f"Unhandled exception: {e}", exc_info=True) 