"""
Configuration utilities for GraphMind
"""

import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional, Union
from omegaconf import OmegaConf, DictConfig
import os


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load configuration from YAML or JSON file
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    if config_path.suffix in ['.yaml', '.yml']:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    elif config_path.suffix == '.json':
        with open(config_path, 'r') as f:
            config = json.load(f)
    else:
        raise ValueError(f"Unsupported config format: {config_path.suffix}")
    
    # Resolve environment variables
    config = resolve_env_vars(config)
    
    return config


def merge_configs(*configs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge multiple configuration dictionaries
    
    Later configs override earlier ones
    """
    merged = {}
    
    for config in configs:
        merged = deep_merge(merged, config)
    
    return merged


def deep_merge(dict1: Dict[str, Any], dict2: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deep merge two dictionaries
    """
    result = dict1.copy()
    
    for key, value in dict2.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    
    return result


def resolve_env_vars(config: Any) -> Any:
    """
    Resolve environment variables in configuration
    
    Supports ${ENV_VAR} and ${ENV_VAR:default} syntax
    """
    if isinstance(config, str):
        # Check for environment variable pattern
        if config.startswith('${') and config.endswith('}'):
            env_var = config[2:-1]
            
            # Check for default value
            if ':' in env_var:
                var_name, default = env_var.split(':', 1)
                return os.environ.get(var_name, default)
            else:
                return os.environ.get(env_var, config)
        return config
    
    elif isinstance(config, dict):
        return {k: resolve_env_vars(v) for k, v in config.items()}
    
    elif isinstance(config, list):
        return [resolve_env_vars(item) for item in config]
    
    return config


def save_config(config: Dict[str, Any], filepath: Union[str, Path]):
    """Save configuration to file"""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    if filepath.suffix in ['.yaml', '.yml']:
        with open(filepath, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    elif filepath.suffix == '.json':
        with open(filepath, 'w') as f:
            json.dump(config, f, indent=2)
    else:
        raise ValueError(f"Unsupported format: {filepath.suffix}")


def create_default_config() -> Dict[str, Any]:
    """Create default GraphMind configuration"""
    return {
        'dataset': {
            'name': 'cora',
            'path': './data',
            'download': True
        },
        'model': {
            'type': 'gcn',
            'hidden_dim': 64,
            'num_layers': 2,
            'dropout': 0.5
        },
        'training': {
            'num_rounds': 100,
            'local_epochs': 1,
            'learning_rate': 0.01,
            'weight_decay': 5e-4
        },
        'consensus': {
            'algorithm': 'ta_bft',
            'byzantine_threshold': 0.33,
            'view_timeout': 10.0
        },
        'partitioner': {
            'objective_weights': {
                'cut': 0.4,
                'balance': 0.4,
                'communication': 0.2
            },
            'coarsening_threshold': 100,
            'refinement_iterations': 10
        },
        'aggregator': {
            'regularization_strength': 0.1,
            'personalization_rate': 0.3
        },
        'privacy': {
            'enabled': True,
            'budget': 1.0
        },
        'logging': {
            'level': 'INFO',
            'structured': True
        }
    }


class ConfigValidator:
    """Validate configuration against schema"""
    
    @staticmethod
    def validate_training_config(config: Dict[str, Any]) -> bool:
        """Validate training configuration"""
        required_keys = ['dataset', 'model', 'training', 'consensus']
        
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Missing required config key: {key}")
        
        # Validate dataset
        if 'name' not in config['dataset']:
            raise ValueError("Dataset name is required")
        
        # Validate model
        if config['model'].get('type') not in ['gcn', 'gat', 'sage']:
            raise ValueError(f"Invalid model type: {config['model'].get('type')}")
        
        # Validate training parameters
        if config['training'].get('num_rounds', 0) <= 0:
            raise ValueError("num_rounds must be positive")
        
        return True