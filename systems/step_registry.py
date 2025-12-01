"""
Step Registry System.

Central registry for all testable steps with metadata, schemas, and validation.
"""

from typing import Dict, Type, List, Optional, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class StepInfo:
    """Metadata for a registered step."""
    name: str
    step_class: Type
    input_schema: Dict[str, Any]
    output_schema: Dict[str, Any]
    parameters: Dict[str, Any]
    description: str = ""
    category: str = "refinement"
    dependencies: List[str] = None
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []


class StepRegistry:
    """
    Central registry for all testable steps.
    
    Provides step discovery, validation, and metadata access.
    """
    
    def __init__(self):
        """Initialize empty registry."""
        self._steps: Dict[str, StepInfo] = {}
        self._decorated_steps: Dict[str, Any] = {}
    
    def register(
        self,
        step_name: str,
        step_class: Type,
        input_schema: Dict[str, Any],
        output_schema: Dict[str, Any],
        parameters: Dict[str, Any],
        description: str = "",
        category: str = "refinement",
        dependencies: Optional[List[str]] = None
    ) -> None:
        """
        Register a step in the registry.
        
        Args:
            step_name: Unique identifier for the step
            step_class: Class that implements the step
            input_schema: Schema defining required inputs (e.g., {"image": np.ndarray, "region": Dict})
            output_schema: Schema defining outputs (e.g., {"angle": float, "rotated_crop": np.ndarray})
            parameters: Parameter definitions (e.g., {"canny_low": int, "canny_high": int})
            description: Human-readable description
            category: Step category (e.g., "refinement", "detection", "extraction")
            dependencies: List of step names this step depends on
        """
        step_info = StepInfo(
            name=step_name,
            step_class=step_class,
            input_schema=input_schema,
            output_schema=output_schema,
            parameters=parameters,
            description=description,
            category=category,
            dependencies=dependencies or []
        )
        
        self._steps[step_name] = step_info
        logger.info(f"Registered step: {step_name} ({category})")
    
    def get_step(self, step_name: str) -> Optional[StepInfo]:
        """
        Get step information by name.
        
        Args:
            step_name: Name of the step
            
        Returns:
            StepInfo if found, None otherwise
        """
        return self._steps.get(step_name)
    
    def list_steps(self, category: Optional[str] = None) -> List[str]:
        """
        List all registered steps, optionally filtered by category.
        
        Args:
            category: Optional category filter
            
        Returns:
            List of step names
        """
        if category:
            return [name for name, info in self._steps.items() if info.category == category]
        return list(self._steps.keys())
    
    def validate_step(self, step_name: str, inputs: Dict[str, Any]) -> bool:
        """
        Validate that inputs match the step's input schema.
        
        Args:
            step_name: Name of the step
            inputs: Input dictionary to validate
            
        Returns:
            True if valid, False otherwise
        """
        step_info = self.get_step(step_name)
        if not step_info:
            logger.error(f"Step not found: {step_name}")
            return False
        
        # Check that all required inputs are present
        for key, expected_type in step_info.input_schema.items():
            if key not in inputs:
                logger.error(f"Missing required input: {key}")
                return False
            
            # Basic type checking (can be enhanced)
            input_value = inputs[key]
            if expected_type is not None:
                # Handle numpy arrays specially
                import numpy as np
                if expected_type == np.ndarray:
                    if not isinstance(input_value, np.ndarray):
                        logger.error(f"Input {key} must be numpy.ndarray, got {type(input_value)}")
                        return False
                elif not isinstance(input_value, expected_type):
                    logger.warning(f"Input {key} type mismatch: expected {expected_type}, got {type(input_value)}")
        
        return True
    
    def get_step_metadata(self, step_name: str) -> Optional[Dict[str, Any]]:
        """
        Get metadata for a step.
        
        Args:
            step_name: Name of the step
            
        Returns:
            Dictionary with step metadata
        """
        step_info = self.get_step(step_name)
        if not step_info:
            return None
        
        return {
            "name": step_info.name,
            "description": step_info.description,
            "category": step_info.category,
            "parameters": step_info.parameters,
            "input_schema": step_info.input_schema,
            "output_schema": step_info.output_schema,
            "dependencies": step_info.dependencies
        }


# Global registry instance
_registry = StepRegistry()


def get_registry() -> StepRegistry:
    """Get the global step registry instance."""
    return _registry


def register_step(
    step_name: str,
    input_schema: Dict[str, Any],
    output_schema: Dict[str, Any],
    parameters: Dict[str, Any],
    description: str = "",
    category: str = "refinement",
    dependencies: Optional[List[str]] = None
):
    """
    Decorator for registering steps.
    
    Usage:
        @register_step(
            "orientation",
            input_schema={"image": np.ndarray, "region": Dict},
            output_schema={"angle": float},
            parameters={"canny_low": int}
        )
        class OrientationStep:
            ...
    """
    def decorator(cls):
        _registry.register(
            step_name=step_name,
            step_class=cls,
            input_schema=input_schema,
            output_schema=output_schema,
            parameters=parameters,
            description=description,
            category=category,
            dependencies=dependencies
        )
        return cls
    return decorator








