"""Market impact models for trading simulation."""
import numpy as np
import logging
import math

logger = logging.getLogger('models.impact')

class AlmgrenChriss:
    """Almgren-Chriss market impact model.
    
    Implements a model to estimate temporary and permanent market impact
    based on the Almgren-Chriss framework.
    """
    
    def __init__(self, volatility=0.02, market_volume=None, participation_rate=0.1, 
                 sigma=0.95, eta=0.0002, gamma=1.5, temporary_impact_factor=0.2, 
                 permanent_impact_factor=0.1):
        """Initialize Almgren-Chriss model parameters.
        
        Args:
            volatility (float): Asset volatility (daily)
            market_volume (float, optional): Daily market volume
            participation_rate (float): Desired participation rate (0.1 = 10%)
            sigma (float): Volatility scaling parameter
            eta (float): Permanent impact factor
            gamma (float): Temporary impact exponent (typically 0.5-1.5)
            temporary_impact_factor (float): Scaling factor for temporary impact
            permanent_impact_factor (float): Scaling factor for permanent impact
        """
        self.volatility = volatility
        self.market_volume = market_volume
        self.participation_rate = participation_rate
        self.sigma = sigma
        self.eta = eta
        self.gamma = gamma
        self.temporary_impact_factor = temporary_impact_factor
        self.permanent_impact_factor = permanent_impact_factor
    
    def estimate_impact(self, price, quantity, orderbook, side="buy"):
        """Estimate market impact using Almgren-Chriss model.
        
        Args:
            price (float): Current market price
            quantity (float): Trade quantity
            orderbook (OrderBook): Current orderbook state
            side (str): Trade side ("buy" or "sell")
            
        Returns:
            dict: Impact estimates
        """
        # Extract orderbook features
        liquidity_factor = self._calculate_liquidity_factor(orderbook, side)
        orderbook_depth = self._calculate_orderbook_depth(orderbook, side)
        
        # If market_volume is not provided, estimate from orderbook
        if self.market_volume is None:
            # Simple estimation from orderbook depth and volatility
            self.market_volume = orderbook_depth * price * (1 + 5 * self.volatility)
        
        # Calculate trade value
        trade_value = price * quantity
        
        # Calculate market parameters
        daily_volatility = self.volatility * price
        
        # Order size as fraction of market volume
        order_pct = quantity / self.market_volume if self.market_volume > 0 else 0.01
        
        # Calculate temporary impact (immediate price slippage)
        # Using a power-law model: impact = factor * (size)^gamma
        temporary_impact_bps = self.temporary_impact_factor * (order_pct ** self.gamma) * 10000  # in basis points
        temporary_impact_pct = temporary_impact_bps / 100  # convert to percentage
        temporary_impact_value = price * (temporary_impact_pct / 100)  # convert to price
        
        # Calculate permanent impact (lasting change in market price)
        # Using linear model: impact = eta * size
        permanent_impact_bps = self.permanent_impact_factor * self.eta * order_pct * 10000  # in basis points
        permanent_impact_pct = permanent_impact_bps / 100  # convert to percentage
        permanent_impact_value = price * (permanent_impact_pct / 100)  # convert to price
        
        # Apply liquidity adjustment
        temporary_impact_value *= (1 + liquidity_factor)
        
        # Apply volatility adjustment
        impact_volatility_factor = 1 + (self.sigma * self.volatility)
        temporary_impact_value *= impact_volatility_factor
        permanent_impact_value *= impact_volatility_factor
        
        # Apply direction
        if side.lower() == "sell":
            temporary_impact_value = -temporary_impact_value
            permanent_impact_value = -permanent_impact_value
        
        # Convert back to percentage
        temporary_impact_pct = (temporary_impact_value / price) * 100
        permanent_impact_pct = (permanent_impact_value / price) * 100
        
        # Total impact is the sum of temporary and permanent
        total_impact_value = temporary_impact_value + permanent_impact_value
        total_impact_pct = (total_impact_value / price) * 100
        
        # Calculate expected execution price after impact
        expected_execution_price = price
        if side.lower() == "buy":
            expected_execution_price += temporary_impact_value
        else:
            expected_execution_price -= temporary_impact_value
        
        logger.debug(f"Market impact estimate: temp={temporary_impact_pct:.6f}%, " 
                    f"perm={permanent_impact_pct:.6f}%, total={total_impact_pct:.6f}%")
                    
        return {
            'temporary_impact_pct': temporary_impact_pct,
            'permanent_impact_pct': permanent_impact_pct,
            'total_impact_pct': total_impact_pct,
            'temporary_impact_value': temporary_impact_value,
            'permanent_impact_value': permanent_impact_value,
            'total_impact_value': total_impact_value,
            'expected_execution_price': expected_execution_price,
            'order_pct_of_volume': order_pct * 100  # as percentage
        }
    
    def _calculate_liquidity_factor(self, orderbook, side):
        """Calculate liquidity factor from orderbook.
        
        Args:
            orderbook (OrderBook): Order book instance
            side (str): Trade side
            
        Returns:
            float: Liquidity factor (0-1)
        """
        # Get relevant side of the book
        if side.lower() == "buy":
            levels = orderbook.get_price_levels('asks', depth=10)
            best_price, _ = orderbook.best_ask() or (None, None)
        else:
            levels = orderbook.get_price_levels('bids', depth=10)
            best_price, _ = orderbook.best_bid() or (None, None)
        
        if not best_price or not levels:
            return 0.5  # default if no data
        
        # Calculate liquidity within 1% of best price
        total_qty = sum(qty for _, qty in levels)
        near_qty = sum(qty for price, qty in levels 
                      if abs(price - best_price) / best_price <= 0.01)
        
        # Liquidity factor (less liquid = higher factor)
        if total_qty > 0:
            liquidity_factor = 1 - (near_qty / total_qty)
        else:
            liquidity_factor = 0.5
        
        return liquidity_factor
    
    def _calculate_orderbook_depth(self, orderbook, side):
        """Calculate total depth of the orderbook.
        
        Args:
            orderbook (OrderBook): Order book instance
            side (str): Trade side
            
        Returns:
            float: Total quantity in the orderbook
        """
        # Get total quantity on relevant side
        if side.lower() == "buy":
            levels = orderbook.get_price_levels('asks', depth=20)
        else:
            levels = orderbook.get_price_levels('bids', depth=20)
        
        if not levels:
            return 100.0  # default if no data
        
        # Sum all quantities
        total_qty = sum(qty for _, qty in levels)
        
        return total_qty
    
    def calculate_optimal_execution_time(self, quantity, price, urgency="medium"):
        """Calculate optimal execution time using Almgren-Chriss framework.
        
        Args:
            quantity (float): Trade quantity
            price (float): Market price
            urgency (str): Execution urgency (low, medium, high)
            
        Returns:
            float: Optimal execution time in hours
        """
        # Urgency parameters (risk aversion)
        urgency_factor = {
            "low": 0.1,
            "medium": 1.0,
            "high": 10.0
        }.get(urgency.lower(), 1.0)
        
        # Simplified optimal time calculation
        # T* = sqrt((sigma^2 * X) / (eta * lambda))
        # where sigma is volatility, X is order size, eta is permanent impact, lambda is risk aversion
        
        # If market_volume is too small, use a minimum
        effective_volume = max(self.market_volume, quantity * 10)
        
        # Order size as fraction of market volume
        order_pct = quantity / effective_volume
        
        # Base execution time (in days)
        base_time = math.sqrt((self.volatility**2 * order_pct) / 
                             (self.eta * urgency_factor))
        
        # Convert to hours (min 0.1 hour, max 48 hours)
        execution_time_hours = min(max(base_time * 24, 0.1), 48)
        
        return execution_time_hours
    
    @classmethod
    def calibrate_from_orderbook(cls, orderbook, volatility=0.02):
        """Create an Almgren-Chriss model calibrated from orderbook data.
        
        Args:
            orderbook (OrderBook): Current orderbook state
            volatility (float): Asset volatility
            
        Returns:
            AlmgrenChriss: Calibrated model instance
        """
        # Extract bid and ask levels
        bid_levels = orderbook.get_price_levels('bids', depth=20)
        ask_levels = orderbook.get_price_levels('asks', depth=20)
        
        # Get best bid and ask
        best_bid, _ = orderbook.best_bid() or (None, None)
        best_ask, _ = orderbook.best_ask() or (None, None)
        
        if not best_bid or not best_ask or not bid_levels or not ask_levels:
            # Default parameters if orderbook is empty
            return cls(volatility=volatility)
        
        # Calculate mid price
        mid_price = (best_bid + best_ask) / 2
        
        # Calculate orderbook depth
        total_bid_qty = sum(qty for _, qty in bid_levels)
        total_ask_qty = sum(qty for _, qty in ask_levels)
        
        # Estimated daily volume (very rough approximation)
        estimated_volume = (total_bid_qty + total_ask_qty) * 1000
        
        # Liquidity within 0.5% of mid price
        near_bid_qty = sum(qty for price, qty in bid_levels 
                          if abs(price - mid_price) / mid_price <= 0.005)
        near_ask_qty = sum(qty for price, qty in ask_levels 
                          if abs(price - mid_price) / mid_price <= 0.005)
        
        # Calculate "tightness" of the book
        if (total_bid_qty + total_ask_qty) > 0:
            tightness = (near_bid_qty + near_ask_qty) / (total_bid_qty + total_ask_qty)
        else:
            tightness = 0.5
        
        # Calibrate model parameters
        eta = 0.0002 * (1 + (1 - tightness))  # permanent impact factor
        gamma = 1.5 * (0.5 + (1 - tightness))  # temporary impact exponent
        
        # Adjust impact factors based on volatility
        temporary_impact_factor = 0.2 * (1 + volatility * 10)
        permanent_impact_factor = 0.1 * (1 + volatility * 5)
        
        return cls(
            volatility=volatility,
            market_volume=estimated_volume,
            eta=eta,
            gamma=gamma,
            temporary_impact_factor=temporary_impact_factor,
            permanent_impact_factor=permanent_impact_factor
        )


def estimate_market_impact(price, quantity, orderbook, side="buy", volatility=0.02):
    """Convenience function to estimate market impact.
    
    Args:
        price (float): Current market price
        quantity (float): Trade quantity
        orderbook (OrderBook): Current orderbook state
        side (str): Trade side ("buy" or "sell")
        volatility (float): Asset volatility
        
    Returns:
        dict: Impact estimates
    """
    # Create and calibrate model
    model = AlmgrenChriss.calibrate_from_orderbook(orderbook, volatility)
    
    # Estimate impact
    return model.estimate_impact(price, quantity, orderbook, side) 