"""
Parameter Management System.

Manages parameter sets, overrides, history, and validation.
"""

import json
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)


class ParameterManager:
    """
    Manages parameter sets, overrides, and history.
    
    Provides loading, saving, merging, and tracking of parameter configurations.
    """
    
    def __init__(self, history_file: Optional[str] = None):
        """
        Initialize parameter manager.
        
        Args:
            history_file: Optional path to history file for tracking
        """
        self.history_file = history_file
        self._history: List[Dict[str, Any]] = []
        if history_file:
            self._load_history()
    
    def load(self, config_path: str) -> Dict[str, Any]:
        """
        Load parameter configuration from file.
        
        Args:
            config_path: Path to config file (JSON or Python module)
            
        Returns:
            Configuration dictionary
        """
        path = Path(config_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        if path.suffix == '.json':
            with open(path, 'r') as f:
                return json.load(f)
        elif path.suffix == '.py':
            # Load Python config module
            import importlib.util
            spec = importlib.util.spec_from_file_location("config", path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Look for REFINEMENT_CONFIG or get_config function
            if hasattr(module, 'REFINEMENT_CONFIG'):
                return module.REFINEMENT_CONFIG.copy()
            elif hasattr(module, 'get_config'):
                return module.get_config()
            else:
                raise ValueError(f"Config file {config_path} has no REFINEMENT_CONFIG or get_config()")
        else:
            raise ValueError(f"Unsupported config file format: {path.suffix}")
    
    def save(self, config_path: str, config: Dict[str, Any]) -> None:
        """
        Save parameter configuration to file.
        
        Args:
            config_path: Path to save config file
            config: Configuration dictionary
        """
        path = Path(config_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Saved config to: {config_path}")
    
    def apply_overrides(
        self,
        base_config: Dict[str, Any],
        overrides: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Apply hierarchical overrides to base configuration.
        
        Supports dot notation: "orientation.canny_low" -> config["orientation"]["canny_low"]
        
        Args:
            base_config: Base configuration dictionary
            overrides: Dictionary of overrides (supports dot notation)
            
        Returns:
            New configuration dictionary with overrides applied
        """
        config = self._deep_copy(base_config)
        
        for key, value in overrides.items():
            keys = key.split('.')
            
            # Navigate nested dict
            d = config
            for k in keys[:-1]:
                if k not in d:
                    d[k] = {}
                d = d[k]
            
            # Set value (try to convert to appropriate type)
            final_key = keys[-1]
            if isinstance(d.get(final_key), (int, float)):
                try:
                    if '.' in str(value):
                        d[final_key] = float(value)
                    else:
                        d[final_key] = int(value)
                except (ValueError, TypeError):
                    d[final_key] = value
            else:
                d[final_key] = value
        
        return config
    
    def merge_configs(self, *configs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge multiple configuration dictionaries.
        
        Later configs override earlier ones.
        
        Args:
            *configs: Configuration dictionaries to merge
            
        Returns:
            Merged configuration dictionary
        """
        merged = {}
        for config in configs:
            merged = self._deep_merge(merged, config)
        return merged
    
    def track_history(self, config: Dict[str, Any], result: Optional[Dict[str, Any]] = None) -> None:
        """
        Track parameter configuration and result in history.
        
        Args:
            config: Configuration dictionary
            result: Optional result dictionary
        """
        entry = {
            'timestamp': datetime.now().isoformat(),
            'config': self._deep_copy(config),
            'result': result
        }
        
        self._history.append(entry)
        
        if self.history_file:
            self._save_history()
    
    def get_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get parameter history.
        
        Args:
            limit: Optional limit on number of entries to return
            
        Returns:
            List of history entries
        """
        if limit:
            return self._history[-limit:]
        return self._history.copy()
    
    def _deep_copy(self, d: Dict[str, Any]) -> Dict[str, Any]:
        """Create deep copy of dictionary."""
        import copy
        return copy.deepcopy(d)
    
    def _deep_merge(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge two dictionaries."""
        result = self._deep_copy(base)
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = self._deep_copy(value)
        
        return result
    
    def _load_history(self) -> None:
        """Load history from file."""
        if not self.history_file:
            return
        
        path = Path(self.history_file)
        if path.exists():
            try:
                with open(path, 'r') as f:
                    self._history = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load history: {e}")
                self._history = []
    
    def _save_history(self) -> None:
        """Save history to file."""
        if not self.history_file:
            return
        
        path = Path(self.history_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(path, 'w') as f:
                json.dump(self._history, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save history: {e}")








