"""
parallel_training.py

Parallel training utilities for classifiers.
Provides multithreaded training of multiple model configurations.

Author: (your name)
"""
from typing import List, Dict, Any, Callable, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading


# -----------------------------
# Thread-safe result collection
# -----------------------------

class ThreadSafeResults:
    """Thread-safe container for collecting training results."""
    
    def __init__(self):
        self.lock = threading.Lock()
        self.results = []
        self.best_accuracy = -1.0
        self.best_result = None
    
    def add_result(self, result: Dict[str, Any]):
        """Add a result and update best if needed."""
        with self.lock:
            self.results.append(result)
            accuracy = result.get('accuracy', -1.0)
            if accuracy > self.best_accuracy:
                self.best_accuracy = accuracy
                self.best_result = result
    
    def get_all_results(self) -> List[Dict[str, Any]]:
        """Get all collected results."""
        with self.lock:
            return self.results.copy()
    
    def get_best(self) -> Optional[Dict[str, Any]]:
        """Get the best result."""
        with self.lock:
            return self.best_result


# -----------------------------
# Parallel training helpers
# -----------------------------

def train_models_parallel(training_func: Callable,
                         param_list: List[Any],
                         max_workers: int) -> List[Dict[str, Any]]:
    """
    Train multiple models in parallel.
    
    Args:
        training_func: Function that takes a parameter and returns a result dict
        param_list: List of parameters to train with
        max_workers: Maximum number of worker threads
        
    Returns:
        List of result dictionaries from training
    """
    results_container = ThreadSafeResults()
    
    def train_and_collect(param):
        """Train model with given parameter and collect result."""
        try:
            result = training_func(param)
            results_container.add_result(result)
            return result
        except Exception as exc:
            print(f"Error training with parameter {param}: {exc}")
            return None
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_param = {
            executor.submit(train_and_collect, param): param 
            for param in param_list
        }
        
        for future in as_completed(future_to_param):
            param = future_to_param[future]
            try:
                future.result()
            except Exception as exc:
                print(f"Parameter {param} generated an exception: {exc}")
    
    return results_container.get_all_results()

