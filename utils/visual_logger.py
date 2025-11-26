"""
Visual logging utilities for debugging detection strategies.
"""

import cv2
import numpy as np
import logging
import os
import json
from typing import Dict, List, Optional
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


class VisualLogger:
    """Manages visual log creation and saving for debugging detection strategies."""
    
    def __init__(self, output_dir: Optional[str] = None):
        """
        Initialize visual logger.
        
        Args:
            output_dir: Base directory for saving logs. If None, uses tests/fixtures/detection_logs/
        """
        self.output_dir = output_dir or 'tests/fixtures/detection_logs'
        self.steps = []
        self.strategy_name = None
        self.image_name = None
    
    def start_log(self, strategy_name: str, image_name: str):
        """Start a new visual log for a strategy."""
        self.strategy_name = strategy_name
        self.image_name = image_name
        self.steps = []
    
    def add_step(
        self,
        step_name: str,
        description: str,
        image: np.ndarray,
        data: Optional[Dict] = None
    ):
        """
        Add a visualization step to the log.
        
        Args:
            step_name: Name of the step (used in filename)
            description: Human-readable description
            image: Annotated image for this step
            data: Additional debug data (coordinates, values, etc.)
        """
        self.steps.append({
            'step_name': step_name,
            'description': description,
            'image': image.copy(),
            'data': data or {}
        })
    
    def save_log(self, final_visualization: np.ndarray) -> str:
        """
        Save visual log to disk.
        
        Args:
            final_visualization: Final annotated result image
            
        Returns:
            Path to saved log directory
        """
        if not self.strategy_name or not self.image_name:
            logger.warning('Cannot save log: strategy_name or image_name not set')
            return ''
        
        # Create directory structure
        # Use full name if it looks like a timestamped run (contains underscore followed by digits)
        # Otherwise use stem to remove file extension
        import re
        if re.search(r'_\d{8}_\d{6}', self.image_name):
            # Timestamped run name, use as-is
            image_base = self.image_name
        else:
            # Regular image name, strip extension
            image_base = Path(self.image_name).stem
        log_dir = Path(self.output_dir) / image_base / self.strategy_name
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Save step images
        for idx, step in enumerate(self.steps):
            step_filename = f'step_{idx:02d}_{step["step_name"]}.jpg'
            step_path = log_dir / step_filename
            cv2.imwrite(str(step_path), step['image'])
        
        # Save final visualization
        final_path = log_dir / 'final_result.jpg'
        cv2.imwrite(str(final_path), final_visualization)
        
        # Save metadata
        metadata = {
            'strategy_name': self.strategy_name,
            'image_name': self.image_name,
            'timestamp': datetime.now().isoformat(),
            'steps': [
                {
                    'step_name': step['step_name'],
                    'description': step['description'],
                    'image_file': f'step_{idx:02d}_{step["step_name"]}.jpg',
                    'data': step['data']
                }
                for idx, step in enumerate(self.steps)
            ],
            'final_image': 'final_result.jpg'
        }
        
        metadata_path = log_dir / 'log.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f'Visual log saved to: {log_dir}')
        return str(log_dir)
    
    def generate_html_index(self, all_logs: List[Dict]) -> str:
        """
        Generate HTML index for browsing all visual logs.
        
        Args:
            all_logs: List of log metadata dictionaries
            
        Returns:
            HTML content as string
        """
        html = """<!DOCTYPE html>
<html>
<head>
    <title>Strip Detection Visual Logs</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .log-entry { border: 1px solid #ddd; margin: 10px 0; padding: 10px; }
        .log-entry h3 { margin-top: 0; }
        .steps { margin-top: 10px; }
        .step { margin: 5px 0; }
        .step img { max-width: 400px; margin: 5px; }
    </style>
</head>
<body>
    <h1>Strip Detection Visual Logs</h1>
"""
        for log in all_logs:
            html += f"""
    <div class="log-entry">
        <h3>{log['strategy_name']} - {log['image_name']}</h3>
        <p>Timestamp: {log.get('timestamp', 'N/A')}</p>
        <div class="steps">
            <h4>Steps:</h4>
"""
            for step in log.get('steps', []):
                html += f"""
            <div class="step">
                <strong>{step['step_name']}</strong>: {step['description']}<br>
                <img src="{step['image_file']}" alt="{step['step_name']}">
            </div>
"""
            html += f"""
            <div class="step">
                <strong>Final Result</strong><br>
                <img src="{log.get('final_image', 'final_result.jpg')}" alt="Final Result">
            </div>
        </div>
    </div>
"""
        html += """
</body>
</html>
"""
        return html



