"""Maker/Taker probability estimation model using logistic regression."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import logging

logger = logging.getLogger('models.maker_taker')

class MakerTakerEstimator:
    """Model to estimate the probability of an order being a maker or taker."""
    
    def __init__(self):
        """Initialize the maker/taker estimator."""
        self.model = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression(solver='liblinear', random_state=42))
        ])
        self.is_trained = False
    
    def extract_features(self, orderbooks, quantities):
        """Extract features from orderbook data for prediction.
        
        Args:
            orderbooks (list): List of OrderBook instances
            quantities (list): List of order quantities
            
        Returns:
            pd.DataFrame: DataFrame with extracted features
        """
        features = []
        
        for i, ob in enumerate(orderbooks):
            if i >= len(quantities):
                break
                
            quantity = quantities[i]
            
            # Get bid and ask levels
            bid_levels = ob.get_price_levels('bids', depth=10)
            ask_levels = ob.get_price_levels('asks', depth=10)
            
            if not bid_levels or not ask_levels:
                continue
                
            # Calculate L2 book imbalance (bids vs asks volume)
            bid_volume = sum(qty for _, qty in bid_levels)
            ask_volume = sum(qty for _, qty in ask_levels)
            
            if bid_volume + ask_volume == 0:
                imbalance = 0
            else:
                imbalance = (bid_volume - ask_volume) / (bid_volume + ask_volume)
            
            # Calculate additional features
            bid_ask_ratio = bid_volume / ask_volume if ask_volume > 0 else 10.0
            
            # Best bid/ask and spread
            best_bid, best_bid_qty = ob.best_bid() or (0, 0)
            best_ask, best_ask_qty = ob.best_ask() or (0, 0)
            
            # Spread features
            spread = ob.spread() or 0
            spread_pct = ob.spread_percentage() or 0
            
            # Calculate relative order size
            rel_size_to_best = quantity / best_ask_qty if best_ask_qty > 0 else 10.0
            rel_size_to_total = quantity / ask_volume if ask_volume > 0 else 10.0
            
            # Combine features
            feature_dict = {
                'imbalance': imbalance,
                'bid_ask_ratio': min(bid_ask_ratio, 10.0),  # Cap extreme values
                'spread': spread,
                'spread_pct': spread_pct,
                'quantity': quantity,
                'rel_size_to_best': min(rel_size_to_best, 10.0),  # Cap extreme values
                'rel_size_to_total': min(rel_size_to_total, 1.0)  # Cap extreme values
            }
            
            features.append(feature_dict)
            
        return pd.DataFrame(features)
    
    def train(self, orderbooks, quantities, maker_taker_labels):
        """Train the logistic regression model.
        
        Args:
            orderbooks (list): List of OrderBook instances
            quantities (list): List of order quantities
            maker_taker_labels (list): Binary labels (1 for maker, 0 for taker)
            
        Returns:
            bool: True if training succeeded
        """
        X = self.extract_features(orderbooks, quantities)
        y = np.array(maker_taker_labels)
        
        if len(X) < 5 or len(X) != len(y):
            logger.warning(f"Insufficient training data: {len(X)} samples, {len(y)} labels")
            return False
        
        # Train the model
        self.model.fit(X, y)
        
        # Get model score
        train_score = self.model.score(X, y)
        logger.info(f"Maker/Taker model trained. Accuracy: {train_score:.4f}")
        
        # Log feature importances if available
        try:
            importances = self.model.named_steps['classifier'].coef_[0]
            feature_importance = pd.DataFrame({
                'Feature': X.columns,
                'Importance': np.abs(importances)
            }).sort_values('Importance', ascending=False)
            
            top_features = feature_importance.head(3)['Feature'].tolist()
            logger.info(f"Top predictive features: {', '.join(top_features)}")
        except (AttributeError, IndexError):
            pass
        
        self.is_trained = True
        return True
    
    def predict_maker_probability(self, orderbook, quantity):
        """Predict the probability of an order being a maker.
        
        Args:
            orderbook (OrderBook): Current orderbook state
            quantity (float): Order quantity
            
        Returns:
            float: Probability of the order being a maker (0.0-1.0)
        """
        if not self.is_trained:
            logger.warning("Model not trained, returning default estimate")
            # Default heuristic based on order size
            relative_size = self._estimate_relative_size(orderbook, quantity)
            return max(0.0, min(0.9, 0.9 - relative_size))
        
        # Extract features for prediction
        X = self.extract_features([orderbook], [quantity])
        
        if X.empty:
            logger.warning("Could not extract features, returning default estimate")
            return 0.8  # Default fallback
        
        # Predict probability
        try:
            # Get probability of class 1 (maker)
            prob = self.model.predict_proba(X)[0][1]
            return float(prob)
        except Exception as e:
            logger.error(f"Error predicting maker probability: {e}")
            return 0.8  # Default fallback
    
    def _estimate_relative_size(self, orderbook, quantity):
        """Estimate relative order size compared to orderbook depth.
        
        Args:
            orderbook (OrderBook): Current orderbook state
            quantity (float): Order quantity
            
        Returns:
            float: Relative size (0.0-1.0)
        """
        # Get ask levels
        ask_levels = orderbook.get_price_levels('asks', depth=10)
        
        if not ask_levels:
            return 0.5  # Default
            
        # Total liquidity in the book
        total_qty = sum(qty for _, qty in ask_levels)
        
        if total_qty == 0:
            return 1.0
            
        # Calculate relative size (capped at 1.0)
        return min(1.0, quantity / total_qty)


def generate_synthetic_training_data(orderbook, num_samples=100):
    """Generate synthetic training data for the maker/taker model.
    
    Args:
        orderbook (OrderBook): Current orderbook state
        num_samples (int): Number of samples to generate
        
    Returns:
        tuple: (orderbooks, quantities, labels)
    """
    orderbooks = [orderbook] * num_samples
    
    # Generate random quantities
    quantities = []
    labels = []
    
    for _ in range(num_samples):
        # Generate quantity - small orders more likely to be makers
        if np.random.random() < 0.7:
            # Small order
            quantity = 0.1 + np.random.exponential(0.5)
            # Small orders more likely to be makers
            maker_prob = 0.8 + (np.random.random() * 0.15)
        else:
            # Large order
            quantity = 1.0 + np.random.exponential(5.0)
            # Large orders more likely to be takers
            maker_prob = 0.3 - (min(1.0, quantity / 10) * 0.3)
        
        quantities.append(quantity)
        
        # Generate label based on probability
        label = 1 if np.random.random() < maker_prob else 0
        labels.append(label)
    
    return orderbooks, quantities, labels


def estimate_maker_taker_ratio(orderbook, quantity, order_type="Limit"):
    """Estimate maker/taker ratio for an order.
    
    Args:
        orderbook (OrderBook): Current orderbook state
        quantity (float): Order quantity
        order_type (str): Order type (Market or Limit)
        
    Returns:
        tuple: (maker_ratio, taker_ratio)
    """
    if order_type.lower() == "market":
        # Market orders are always taker
        return 0.0, 1.0
    
    # For limit orders, create and train a model
    model = MakerTakerEstimator()
    
    # Generate synthetic training data
    obs, qtys, labels = generate_synthetic_training_data(orderbook)
    
    # Train the model
    model.train(obs, qtys, labels)
    
    # Predict maker probability
    maker_prob = model.predict_maker_probability(orderbook, quantity)
    
    # Return as ratio
    return maker_prob, 1.0 - maker_prob 