"""Fee calculation module for crypto exchanges."""
import logging
from models.maker_taker import estimate_maker_taker_ratio

logger = logging.getLogger('utils.fees')

# OKX fee tier structure (as of 2023)
# Format: (maker_fee, taker_fee) in percentage
OKX_FEE_TIERS = {
    'Regular': (0.080, 0.100),  # 0.08% maker, 0.10% taker
    'VIP 1': (0.060, 0.080),    # 0.06% maker, 0.08% taker
    'VIP 2': (0.050, 0.070),    # 0.05% maker, 0.07% taker
    'VIP 3': (0.040, 0.060),    # 0.04% maker, 0.06% taker
    'VIP 4': (0.020, 0.050),    # 0.02% maker, 0.05% taker
    'VIP 5': (0.015, 0.045),    # 0.015% maker, 0.045% taker
}

def calculate_fee(executed_price, quantity, fee_tier='Regular', order_type='Market', orderbook=None):
    """Calculate trading fee based on price, quantity, fee tier and order type.
    
    Args:
        executed_price (float): Execution price of the trade
        quantity (float): Quantity traded
        fee_tier (str): Fee tier (Regular, VIP 1, VIP 2, etc.)
        order_type (str): Order type (Market or Limit)
        orderbook (OrderBook, optional): Current orderbook state for maker/taker estimation
        
    Returns:
        tuple: (fee_amount, fee_rate, maker_taker_ratio)
    """
    # Get fee rates for the specified tier
    maker_rate, taker_rate = OKX_FEE_TIERS.get(fee_tier, OKX_FEE_TIERS['Regular'])
    
    # Convert percentage to decimal
    maker_rate_decimal = maker_rate / 100
    taker_rate_decimal = taker_rate / 100
    
    # Determine maker/taker ratio based on order type
    if order_type.lower() == 'market':
        # Market orders are always taker
        maker_ratio = 0.0
        taker_ratio = 1.0
        effective_rate = taker_rate_decimal
    else:
        # For limit orders, estimate maker/taker ratio using our model
        if orderbook:
            maker_ratio, taker_ratio = estimate_maker_taker_ratio(orderbook, quantity, order_type)
        else:
            # Fallback to simplified model if no orderbook provided
            maker_ratio = 0.8
            taker_ratio = 0.2
            
        effective_rate = (maker_rate_decimal * maker_ratio) + (taker_rate_decimal * taker_ratio)
    
    # Calculate fee amount
    trade_value = executed_price * quantity
    fee_amount = trade_value * effective_rate
    
    # Format maker/taker ratio as a string
    maker_taker_ratio = f"{int(maker_ratio * 100)}/{int(taker_ratio * 100)}%"
    
    logger.debug(f"Fee calculation: {fee_tier} tier, {order_type} order, " 
                f"rate: {effective_rate*100:.4f}%, amount: {fee_amount:.6f}")
    
    return {
        'fee_amount': fee_amount,
        'fee_rate': effective_rate,
        'fee_rate_percentage': effective_rate * 100,
        'maker_taker_ratio': maker_taker_ratio,
        'maker_rate': maker_rate_decimal,
        'taker_rate': taker_rate_decimal
    }

def get_fee_rate(fee_tier='Regular', is_maker=False):
    """Get the fee rate for a given tier and maker/taker status.
    
    Args:
        fee_tier (str): Fee tier (Regular, VIP 1, VIP 2, etc.)
        is_maker (bool): Whether the trade is a maker (True) or taker (False)
        
    Returns:
        float: Fee rate as a decimal (not percentage)
    """
    maker_rate, taker_rate = OKX_FEE_TIERS.get(fee_tier, OKX_FEE_TIERS['Regular'])
    
    # Convert percentage to decimal
    if is_maker:
        return maker_rate / 100
    else:
        return taker_rate / 100

def estimate_fee_breakdown(trade_value, fee_tier='Regular', order_type='Market', orderbook=None, quantity=None):
    """Provide a detailed breakdown of fees.
    
    Args:
        trade_value (float): Total value of the trade
        fee_tier (str): Fee tier (Regular, VIP 1, VIP 2, etc.)
        order_type (str): Order type (Market or Limit)
        orderbook (OrderBook, optional): Current orderbook state for maker/taker estimation
        quantity (float, optional): Order quantity for maker/taker estimation
        
    Returns:
        dict: Fee breakdown details
    """
    maker_rate, taker_rate = OKX_FEE_TIERS.get(fee_tier, OKX_FEE_TIERS['Regular'])
    
    # Determine maker/taker ratio
    if order_type.lower() == 'market':
        maker_ratio = 0.0
        taker_ratio = 1.0
    else:
        # For limit orders, estimate maker/taker ratio using our model
        if orderbook and quantity:
            maker_ratio, taker_ratio = estimate_maker_taker_ratio(orderbook, quantity, order_type)
        else:
            # Fallback to simplified model if no orderbook or quantity provided
            maker_ratio = 0.8
            taker_ratio = 0.2
    
    # Calculate trade amounts for maker and taker
    maker_amount = trade_value * maker_ratio
    taker_amount = trade_value * taker_ratio
    
    # Calculate fee components
    maker_fee = maker_amount * (maker_rate / 100)
    taker_fee = taker_amount * (taker_rate / 100)
    total_fee = maker_fee + taker_fee
    
    return {
        'maker_amount': maker_amount,
        'taker_amount': taker_amount,
        'maker_fee': maker_fee,
        'taker_fee': taker_fee,
        'total_fee': total_fee,
        'effective_rate': (total_fee / trade_value) * 100,  # as percentage
        'maker_ratio': maker_ratio,
        'taker_ratio': taker_ratio
    } 