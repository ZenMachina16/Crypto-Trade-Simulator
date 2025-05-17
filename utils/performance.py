"""Performance tracking and latency measurement utilities."""
import time
import logging
import threading
import statistics
from collections import deque
import numpy as np

logger = logging.getLogger('utils.performance')

class LatencyTracker:
    """Tracks latency statistics for system monitoring."""
    
    def __init__(self, name, max_samples=1000, log_interval=10):
        """Initialize the latency tracker.
        
        Args:
            name (str): Name of the component being tracked
            max_samples (int): Maximum number of samples to keep in memory
            log_interval (int): Interval in seconds for logging statistics
        """
        self.name = name
        self.max_samples = max_samples
        self.log_interval = log_interval
        self.latencies = deque(maxlen=max_samples)
        self.start_times = {}
        self.last_log_time = time.time()
        self.lock = threading.Lock()
        self.is_logging_active = False
        self.logging_thread = None
    
    def start(self, event_id=None):
        """Start timing an event.
        
        Args:
            event_id: Optional identifier for the event (default: current time)
            
        Returns:
            event_id: The event identifier
        """
        if event_id is None:
            event_id = time.time_ns()
            
        self.start_times[event_id] = time.time()
        return event_id
    
    def stop(self, event_id):
        """Stop timing an event and record the latency.
        
        Args:
            event_id: Identifier returned from start()
            
        Returns:
            float: The measured latency in milliseconds
        """
        if event_id not in self.start_times:
            return None
            
        end_time = time.time()
        start_time = self.start_times.pop(event_id)
        latency_ms = (end_time - start_time) * 1000  # Convert to milliseconds
        
        with self.lock:
            self.latencies.append(latency_ms)
            
        # Check if it's time to log statistics
        current_time = time.time()
        if current_time - self.last_log_time >= self.log_interval:
            self._log_statistics()
            self.last_log_time = current_time
            
        return latency_ms
    
    def _log_statistics(self):
        """Log latency statistics."""
        with self.lock:
            if not self.latencies:
                return
                
            avg_latency = statistics.mean(self.latencies)
            max_latency = max(self.latencies)
            min_latency = min(self.latencies)
            
            # Calculate percentiles if we have enough samples
            if len(self.latencies) >= 10:
                p50 = np.percentile(self.latencies, 50)
                p95 = np.percentile(self.latencies, 95)
                p99 = np.percentile(self.latencies, 99)
                
                logger.info(f"Latency stats for {self.name} (ms) - "
                           f"Avg: {avg_latency:.2f}, Min: {min_latency:.2f}, Max: {max_latency:.2f}, "
                           f"p50: {p50:.2f}, p95: {p95:.2f}, p99: {p99:.2f}, "
                           f"Samples: {len(self.latencies)}")
            else:
                logger.info(f"Latency stats for {self.name} (ms) - "
                           f"Avg: {avg_latency:.2f}, Min: {min_latency:.2f}, Max: {max_latency:.2f}, "
                           f"Samples: {len(self.latencies)}")
    
    def start_periodic_logging(self):
        """Start a background thread to log statistics periodically."""
        if self.is_logging_active:
            return
            
        self.is_logging_active = True
        
        def logging_worker():
            while self.is_logging_active:
                time.sleep(self.log_interval)
                self._log_statistics()
                
        self.logging_thread = threading.Thread(target=logging_worker, daemon=True)
        self.logging_thread.start()
        
    def stop_periodic_logging(self):
        """Stop the periodic logging thread."""
        self.is_logging_active = False
        if self.logging_thread and self.logging_thread.is_alive():
            self.logging_thread.join(timeout=1.0)
    
    def reset(self):
        """Clear all recorded latencies."""
        with self.lock:
            self.latencies.clear()
            self.start_times.clear()
    
    def get_statistics(self):
        """Get current latency statistics.
        
        Returns:
            dict: Dictionary with latency statistics
        """
        with self.lock:
            if not self.latencies:
                return {
                    'avg': 0,
                    'min': 0,
                    'max': 0,
                    'p50': 0,
                    'p95': 0,
                    'p99': 0,
                    'samples': 0
                }
                
            stats = {
                'avg': statistics.mean(self.latencies),
                'min': min(self.latencies),
                'max': max(self.latencies),
                'samples': len(self.latencies)
            }
            
            # Calculate percentiles if we have enough samples
            if len(self.latencies) >= 10:
                stats['p50'] = np.percentile(self.latencies, 50)
                stats['p95'] = np.percentile(self.latencies, 95)
                stats['p99'] = np.percentile(self.latencies, 99)
            else:
                stats['p50'] = stats['avg']
                stats['p95'] = stats['max']
                stats['p99'] = stats['max']
                
            return stats


# Global latency trackers
ui_update_tracker = LatencyTracker("UI Update", log_interval=10)
order_processing_tracker = LatencyTracker("Order Processing", log_interval=10)
market_data_tracker = LatencyTracker("Market Data", log_interval=10) 