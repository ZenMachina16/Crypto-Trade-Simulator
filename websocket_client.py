#!/usr/bin/env python3
import asyncio
import json
import websockets
from models.orderbook import OrderBook

async def connect_to_orderbook():
    uri = "wss://ws.gomarket-cpp.goquant.io/ws/l2-orderbook/okx/BTC-USDT-SWAP"
    symbol = "BTC-USDT-SWAP"
    orderbook = OrderBook(symbol)
    
    print(f"Connecting to {uri}...")
    
    async with websockets.connect(uri) as websocket:
        print(f"Connection established for {symbol}!")
        
        while True:
            try:
                # Receive message from the WebSocket
                message = await websocket.recv()
                
                # Parse JSON payload to Python dict
                data = json.loads(message)
                
                # Update the orderbook with the new data
                orderbook.update(data)
                
                # Get best bid and best ask
                bid_price, bid_qty = orderbook.best_bid() or (None, None)
                ask_price, ask_qty = orderbook.best_ask() or (None, None)
                
                # Print best bid and ask
                if bid_price and ask_price:
                    print(f"Best Bid: {bid_price:.2f} @ {bid_qty:.6f} | Best Ask: {ask_price:.2f} @ {ask_qty:.6f} | Spread: {ask_price - bid_price:.2f}")
                
            except websockets.exceptions.ConnectionClosed:
                print("Connection closed, attempting to reconnect...")
                break
            except Exception as e:
                print(f"Error: {e}")
                break

async def main():
    while True:
        try:
            await connect_to_orderbook()
        except Exception as e:
            print(f"Connection error: {e}")
        
        print("Reconnecting in 5 seconds...")
        await asyncio.sleep(5)

if __name__ == "__main__":
    asyncio.run(main()) 