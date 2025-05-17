#!/usr/bin/env python3
"""Main entry point for the Crypto Trade Simulator."""
import asyncio
import logging
import argparse
from websocket.client import OrderBookClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('main')

async def run_orderbook_client(symbol):
    """Run the order book client for a specific symbol.
    
    Args:
        symbol (str): Trading pair symbol
    """
    # L2 Order Book WebSocket URL template
    uri = f"wss://ws.gomarket-cpp.goquant.io/ws/l2-orderbook/okx/{symbol}"
    
    client = OrderBookClient(uri, symbol)
    await client.connect()

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Crypto Trade Simulator')
    parser.add_argument('--symbol', type=str, default='BTC-USDT-SWAP',
                        help='Trading pair symbol (default: BTC-USDT-SWAP)')
    return parser.parse_args()

def main():
    """Main entry point for the application."""
    args = parse_args()
    
    try:
        asyncio.run(run_orderbook_client(args.symbol))
    except KeyboardInterrupt:
        logger.info("Program interrupted by user. Exiting...")
    except Exception as e:
        logger.error(f"Unhandled exception: {e}", exc_info=True)

if __name__ == "__main__":
    main() 