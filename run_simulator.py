#!/usr/bin/env python3
"""Run the crypto trade simulator."""
import logging
import atexit
from ui.simulator import create_simulator_ui
from utils.performance import ui_update_tracker, market_data_tracker, order_processing_tracker

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('simulator')

def start_performance_tracking():
    """Initialize performance tracking."""
    logger.info("Starting performance tracking")
    ui_update_tracker.start_periodic_logging()
    market_data_tracker.start_periodic_logging()
    order_processing_tracker.start_periodic_logging()

def cleanup():
    """Clean up resources on exit."""
    logger.info("Shutting down performance tracking")
    ui_update_tracker.stop_periodic_logging()
    market_data_tracker.stop_periodic_logging()
    order_processing_tracker.stop_periodic_logging()
    
    # Log final statistics
    logger.info("Final UI update latency stats:")
    ui_update_tracker._log_statistics()
    
    logger.info("Final market data latency stats:")
    market_data_tracker._log_statistics()
    
    logger.info("Final order processing latency stats:")
    order_processing_tracker._log_statistics()

def main():
    """Run the trade simulator UI."""
    logger.info("Starting Crypto Trade Simulator")
    
    # Initialize performance tracking
    start_performance_tracking()
    
    # Register cleanup handler
    atexit.register(cleanup)
    
    # Start the UI
    root, simulator = create_simulator_ui()
    root.mainloop()
    
    logger.info("Simulator closed")

if __name__ == "__main__":
    main() 