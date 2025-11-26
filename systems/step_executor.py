"""
Step Execution System.

Executes steps in isolation with validation, parameter injection, and debug support.
"""

import logging
from typing import Dict, Any, Optional
import numpy as np

from systems.step_registry import StepRegistry, get_registry
from services.utils.debug import DebugContext

logger = logging.getLogger(__name__)


class StepExecutor:
    """
    Executes steps in isolation with validation and debug support.
    
    Provides a clean interface for testing individual steps without
    running the full pipeline.
    """
    
    def __init__(self, registry: Optional[StepRegistry] = None):
        """
        Initialize step executor.
        
        Args:
            registry: Step registry to use (defaults to global registry)
        """
        self.registry = registry or get_registry()
    
    def execute(
        self,
        step_name: str,
        inputs: Dict[str, Any],
        parameters: Dict[str, Any],
        debug: Optional[DebugContext] = None
    ) -> Dict[str, Any]:
        """
        Execute a step with given inputs and parameters.
        
        Args:
            step_name: Name of the step to execute
            inputs: Input dictionary (must match step's input schema)
            parameters: Parameter dictionary (step-specific)
            debug: Optional debug context for logging
            
        Returns:
            Dictionary with step output (matches output schema)
            
        Raises:
            ValueError: If step not found or inputs invalid
            RuntimeError: If step execution fails
        """
        # Get step info
        step_info = self.registry.get_step(step_name)
        if not step_info:
            raise ValueError(f"Step not found: {step_name}")
        
        # Validate inputs
        if not self.registry.validate_step(step_name, inputs):
            raise ValueError(f"Invalid inputs for step: {step_name}")
        
        # Create step instance
        try:
            step_instance = step_info.step_class()
            
            # Inject parameters if step accepts config
            if hasattr(step_instance, 'config'):
                # Merge parameters into config
                if isinstance(step_instance.config, dict):
                    step_instance.config.update(parameters)
                else:
                    step_instance.config = parameters
            
            # Execute step
            # Steps should have a refine() or execute() method
            if hasattr(step_instance, 'refine'):
                # Refinement steps use refine(image, input_region, debug)
                if 'image' in inputs and 'region' in inputs:
                    result = step_instance.refine(
                        inputs['image'],
                        inputs['region'],
                        debug
                    )
                    # Handle tuple return (region, metadata)
                    if isinstance(result, tuple):
                        region, metadata = result
                        return {
                            'region': region,
                            'metadata': metadata
                        }
                    return result
                else:
                    raise ValueError(f"Step {step_name} requires 'image' and 'region' inputs")
            
            elif hasattr(step_instance, 'execute'):
                # Generic execute method
                result = step_instance.execute(inputs, parameters, debug)
                return result
            
            else:
                raise RuntimeError(f"Step {step_name} has no execute() or refine() method")
        
        except Exception as e:
            logger.error(f"Step execution failed: {step_name}", exc_info=True)
            raise RuntimeError(f"Step execution failed: {e}") from e
    
    def validate_inputs(self, step_name: str, inputs: Dict[str, Any]) -> bool:
        """
        Validate inputs against step schema.
        
        Args:
            step_name: Name of the step
            inputs: Input dictionary to validate
            
        Returns:
            True if valid, False otherwise
        """
        return self.registry.validate_step(step_name, inputs)
    
    def get_step_info(self, step_name: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a step.
        
        Args:
            step_name: Name of the step
            
        Returns:
            Dictionary with step metadata, or None if not found
        """
        return self.registry.get_step_metadata(step_name)

