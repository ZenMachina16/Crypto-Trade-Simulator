"""Slippage estimation models using regression techniques."""
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import statsmodels.api as sm
import statsmodels.formula.api as smf
import logging
import pandas as pd

logger = logging.getLogger('models.slippage')

class SlippageEstimator:
    """Class for estimating slippage based on order quantity and orderbook depth."""
    
    def __init__(self):
        """Initialize slippage estimator models."""
        self.linear_model = None
        self.quantile_model = None
        self.poly_features = PolynomialFeatures(degree=2, include_bias=False)
        self.is_trained = False
    
    def _extract_features(self, orderbooks, side='buy'):
        """Extract features from orderbook data for regression model.
        
        Args:
            orderbooks (list): List of OrderBook instances
            side (str or list): Trade side, 'buy' or 'sell', or list of sides
            
        Returns:
            pd.DataFrame: DataFrame with orderbook features
        """
        features = []
        
        # Handle both string and list inputs for 'side'
        if isinstance(side, list):
            sides = side
        else:
            sides = [side] * len(orderbooks)
        
        for i, ob in enumerate(orderbooks):
            # Make sure we have a corresponding side for this orderbook
            if i >= len(sides):
                break
                
            current_side = sides[i].lower() if isinstance(sides[i], str) else 'buy'
            
            # Extract top levels from the orderbook
            if current_side == 'buy':
                levels = ob.get_price_levels('asks', depth=5)
            else:
                levels = ob.get_price_levels('bids', depth=5)
            
            # Get best price level
            if current_side == 'buy':
                best_price, _ = ob.best_ask() or (None, None)
            else:
                best_price, _ = ob.best_bid() or (None, None)
                
            if not best_price or not levels:
                continue
                
            # Feature extraction: price, quantity, and cumulative values
            total_qty = sum(qty for _, qty in levels)
            weighted_price = sum(price * qty for price, qty in levels) / total_qty if total_qty > 0 else 0
            price_range = max(price for price, _ in levels) - min(price for price, _ in levels)
            
            # Calculate liquidity measures
            liquidity_within_1pct = sum(qty for price, qty in levels 
                                       if abs(price - best_price) / best_price <= 0.01)
                
            feature_dict = {
                'best_price': best_price,
                'total_quantity': total_qty,
                'weighted_price': weighted_price,
                'price_range': price_range,
                'liquidity_within_1pct': liquidity_within_1pct,
                'spread': ob.spread() or 0,
                'spread_pct': ob.spread_percentage() or 0,
                'book_depth': len(levels)
            }
            
            features.append(feature_dict)
            
        return pd.DataFrame(features)
    
    def _process_trade_data(self, trades, orderbooks):
        """Process trade data to create training dataset.
        
        Args:
            trades (list): List of trade data (qty, executed_price, market_price)
            orderbooks (list): Corresponding orderbook snapshots
            
        Returns:
            tuple: (X_features, y_slippage) for model training
        """
        # Extract sides from the trades
        sides = [t['side'] for t in trades]
        
        # Pass the list of sides to the feature extraction method
        features_df = self._extract_features(orderbooks, sides)
        
        # Calculate actual slippage for each trade
        slippage = []
        quantities = []
        
        for trade in trades:
            executed_price = trade['executed_price']
            market_price = trade['market_price']
            quantity = trade['quantity']
            side = trade['side']
            
            # Calculate slippage percentage
            if side.lower() == 'buy':
                # For buys, slippage is positive when execution price is higher than market
                slip_pct = (executed_price - market_price) / market_price * 100
            else:
                # For sells, slippage is positive when execution price is lower than market
                slip_pct = (market_price - executed_price) / market_price * 100
                
            slippage.append(slip_pct)
            quantities.append(quantity)
            
        # Add quantity to features
        features_df['quantity'] = quantities
        
        return features_df, np.array(slippage)
    
    def train_linear_model(self, trades, orderbooks):
        """Train linear regression model for slippage estimation.
        
        Args:
            trades (list): List of trade data
            orderbooks (list): Corresponding orderbook snapshots
        """
        X, y = self._process_trade_data(trades, orderbooks)
        
        if len(X) < 2:
            logger.warning("Not enough data to train the model")
            return False
            
        # Create polynomial features
        X_poly = self.poly_features.fit_transform(X.values)
        
        # Train linear regression model
        self.linear_model = LinearRegression()
        self.linear_model.fit(X_poly, y)
        
        # Log model performance
        y_pred = self.linear_model.predict(X_poly)
        mse = np.mean((y_pred - y) ** 2)
        logger.info(f"Linear model trained. MSE: {mse:.6f}")
        
        self.is_trained = True
        return True
    
    def train_quantile_model(self, trades, orderbooks, quantiles=[0.25, 0.5, 0.75]):
        """Train quantile regression model for slippage estimation.
        
        Args:
            trades (list): List of trade data
            orderbooks (list): Corresponding orderbook snapshots
            quantiles (list): Quantiles to estimate
        """
        X, y = self._process_trade_data(trades, orderbooks)
        
        if len(X) < 2:
            logger.warning("Not enough data to train the model")
            return False
        
        # Prepare data for statsmodels
        data = X.copy()
        data['slippage'] = y
        
        # Define formula based on features
        formula = 'slippage ~ ' + ' + '.join(X.columns)
        
        # Train quantile regression models for different quantiles
        self.quantile_models = {}
        
        for q in quantiles:
            model = smf.quantreg(formula=formula, data=data)
            result = model.fit(q=q)
            self.quantile_models[q] = result
            
            # Log model performance
            logger.info(f"Quantile {q} model trained. Pseudo R-squared: {result.prsquared:.6f}")
        
        self.is_trained = True
        return True
    
    def estimate_slippage_linear(self, quantity, orderbook, side='buy'):
        """Estimate slippage using the linear model.
        
        Args:
            quantity (float): Order quantity
            orderbook (OrderBook): Current orderbook
            side (str): Trade side, 'buy' or 'sell'
            
        Returns:
            float: Estimated slippage percentage
        """
        if not self.is_trained or not self.linear_model:
            logger.warning("Model is not trained yet")
            return self._simple_slippage_estimate(quantity, orderbook, side)
        
        # Extract features from the current orderbook
        features_df = self._extract_features([orderbook], side)
        if features_df.empty:
            logger.warning("Could not extract features from orderbook")
            return self._simple_slippage_estimate(quantity, orderbook, side)
        
        # Add quantity to features
        features_df['quantity'] = quantity
        
        # Create polynomial features
        X_poly = self.poly_features.transform(features_df.values)
        
        # Predict slippage
        estimated_slippage = self.linear_model.predict(X_poly)[0]
        
        # Apply sanity limits
        estimated_slippage = max(0, min(estimated_slippage, 10.0))
        
        return estimated_slippage
    
    def estimate_slippage_quantile(self, quantity, orderbook, side='buy', quantile=0.5):
        """Estimate slippage using the quantile regression model.
        
        Args:
            quantity (float): Order quantity
            orderbook (OrderBook): Current orderbook
            side (str): Trade side, 'buy' or 'sell'
            quantile (float): Quantile to estimate (0.5 for median)
            
        Returns:
            float: Estimated slippage percentage for the given quantile
        """
        if not self.is_trained or not self.quantile_models or quantile not in self.quantile_models:
            logger.warning(f"Model for quantile {quantile} is not trained yet")
            return self._simple_slippage_estimate(quantity, orderbook, side)
        
        # Extract features from the current orderbook
        features_df = self._extract_features([orderbook], side)
        if features_df.empty:
            logger.warning("Could not extract features from orderbook")
            return self._simple_slippage_estimate(quantity, orderbook, side)
        
        # Add quantity to features
        features_df['quantity'] = quantity
        
        # Prepare data for prediction
        data = features_df.copy()
        data['slippage'] = 0  # Dummy value, not used for prediction
        
        # Predict slippage
        model = self.quantile_models[quantile]
        estimated_slippage = model.predict(data).values[0]
        
        # Apply sanity limits
        estimated_slippage = max(0, min(estimated_slippage, 10.0))
        
        return estimated_slippage
    
    def _simple_slippage_estimate(self, quantity, orderbook, side='buy'):
        """Provide a simple slippage estimate based on order book depth.
        Used as a fallback when models aren't trained.
        
        Args:
            quantity (float): Order quantity
            orderbook (OrderBook): Current orderbook
            side (str): Trade side, 'buy' or 'sell'
            
        Returns:
            float: Simple slippage estimate
        """
        if side.lower() == 'buy':
            levels = orderbook.get_price_levels('asks', depth=10)
        else:
            levels = orderbook.get_price_levels('bids', depth=10)
            
        if not levels:
            return 0.0
            
        # Get best price as reference
        if side.lower() == 'buy':
            best_price, _ = orderbook.best_ask() or (None, None)
        else:
            best_price, _ = orderbook.best_bid() or (None, None)
            
        if not best_price:
            return 0.0
            
        # Calculate weighted average price for the quantity
        remaining_qty = quantity
        executed_value = 0
        
        for price, available_qty in levels:
            if remaining_qty <= 0:
                break
                
            executed_qty = min(remaining_qty, available_qty)
            executed_value += price * executed_qty
            remaining_qty -= executed_qty
            
        if quantity - remaining_qty <= 0:
            return 0.0  # Not enough liquidity in the orderbook
            
        avg_price = executed_value / (quantity - remaining_qty)
        
        # Calculate slippage percentage
        if side.lower() == 'buy':
            slippage = (avg_price - best_price) / best_price * 100
        else:
            slippage = (best_price - avg_price) / best_price * 100
            
        return max(0, slippage)


# Example usage
def estimate_trade_cost(quantity, orderbook, side='buy', estimator=None):
    """Estimate the total cost of a trade including slippage.
    
    Args:
        quantity (float): Order quantity
        orderbook (OrderBook): Current orderbook
        side (str): Trade side, 'buy' or 'sell'
        estimator (SlippageEstimator, optional): Trained estimator
        
    Returns:
        dict: Trade cost estimates
    """
    # Get best bid/ask
    if side.lower() == 'buy':
        best_price, _ = orderbook.best_ask() or (None, None)
    else:
        best_price, _ = orderbook.best_bid() or (None, None)
        
    if not best_price:
        return {
            'market_price': None,
            'estimated_execution_price': None,
            'slippage_pct': None,
            'total_cost': None
        }
        
    # Estimate slippage using the model if available
    if estimator and estimator.is_trained:
        # Get median, 25th and 75th percentile estimates
        if hasattr(estimator, 'quantile_models') and estimator.quantile_models:
            slippage_pct = estimator.estimate_slippage_quantile(quantity, orderbook, side, 0.5)
            slippage_lower = estimator.estimate_slippage_quantile(quantity, orderbook, side, 0.25)
            slippage_upper = estimator.estimate_slippage_quantile(quantity, orderbook, side, 0.75)
        else:
            slippage_pct = estimator.estimate_slippage_linear(quantity, orderbook, side)
            # Approximate confidence interval
            slippage_lower = max(0, slippage_pct * 0.8)
            slippage_upper = slippage_pct * 1.2
    else:
        # Use simple estimate
        estimator = SlippageEstimator()
        slippage_pct = estimator._simple_slippage_estimate(quantity, orderbook, side)
        slippage_lower = max(0, slippage_pct * 0.8)
        slippage_upper = slippage_pct * 1.2
        
    # Calculate estimated execution price
    if side.lower() == 'buy':
        execution_price = best_price * (1 + slippage_pct/100)
        execution_price_lower = best_price * (1 + slippage_lower/100)
        execution_price_upper = best_price * (1 + slippage_upper/100)
    else:
        execution_price = best_price * (1 - slippage_pct/100)
        execution_price_lower = best_price * (1 - slippage_upper/100)  # Note: upper bound gives lower price
        execution_price_upper = best_price * (1 - slippage_lower/100)
        
    # Calculate total cost
    total_cost = execution_price * quantity
    total_cost_range = (execution_price_lower * quantity, execution_price_upper * quantity)
    
    return {
        'market_price': best_price,
        'estimated_execution_price': execution_price,
        'execution_price_range': (execution_price_lower, execution_price_upper),
        'slippage_pct': slippage_pct,
        'slippage_range': (slippage_lower, slippage_upper),
        'total_cost': total_cost,
        'total_cost_range': total_cost_range
    } 