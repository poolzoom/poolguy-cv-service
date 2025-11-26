"""
Comparison Engine System.

Compares results from multiple runs and generates comparison reports.
"""

import json
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path
from dataclasses import dataclass
import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class RunResult:
    """Result from a single test run."""
    run_path: str
    image_name: str
    parameters: Dict[str, Any]
    steps: List[Dict[str, Any]]
    log_path: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class ComparisonResult:
    """Result of comparing two runs."""
    run1: RunResult
    run2: RunResult
    differences: Dict[str, Any]
    visual_diff_path: Optional[str] = None


class ComparisonEngine:
    """
    Compares results from multiple runs and generates reports.
    
    Provides side-by-side comparison, metrics comparison, and visual diffs.
    """
    
    def __init__(self):
        """Initialize comparison engine."""
        pass
    
    def load_run(self, run_path: str) -> RunResult:
        """
        Load a run result from debug log directory.
        
        Args:
            run_path: Path to run directory (contains log.json)
            
        Returns:
            RunResult object
        """
        path = Path(run_path)
        log_file = path / 'log.json'
        
        if not log_file.exists():
            raise FileNotFoundError(f"Log file not found: {log_file}")
        
        with open(log_file, 'r') as f:
            log_data = json.load(f)
        
        # Extract image name from path or log
        image_name = log_data.get('image_name', Path(run_path).parent.name)
        
        # Extract parameters from log
        parameters = {}
        for step in log_data.get('steps', []):
            step_data = step.get('data', {})
            # Look for parameter-like keys
            for key in ['canny_low', 'canny_high', 'hough_threshold', 'rotation_angle']:
                if key in step_data:
                    parameters[key] = step_data[key]
        
        return RunResult(
            run_path=str(path),
            image_name=image_name,
            parameters=parameters,
            steps=log_data.get('steps', []),
            log_path=str(log_file),
            metadata={
                'timestamp': log_data.get('timestamp'),
                'strategy_name': log_data.get('strategy_name'),
                'step_count': len(log_data.get('steps', []))
            }
        )
    
    def compare(
        self,
        run1: RunResult,
        run2: RunResult,
        comparison_type: str = "side_by_side"
    ) -> ComparisonResult:
        """
        Compare two run results.
        
        Args:
            run1: First run result
            run2: Second run result
            comparison_type: Type of comparison ("side_by_side", "metrics", "visual")
            
        Returns:
            ComparisonResult with differences
        """
        differences = {
            'parameters': self._compare_parameters(run1.parameters, run2.parameters),
            'steps': self._compare_steps(run1.steps, run2.steps),
            'metadata': self._compare_metadata(run1.metadata, run2.metadata)
        }
        
        return ComparisonResult(
            run1=run1,
            run2=run2,
            differences=differences
        )
    
    def visualize_differences(
        self,
        run1: RunResult,
        run2: RunResult,
        output_path: Optional[str] = None
    ) -> np.ndarray:
        """
        Create visual comparison image showing differences.
        
        Args:
            run1: First run result
            run2: Second run result
            output_path: Optional path to save comparison image
            
        Returns:
            Comparison image as numpy array
        """
        # Load final images from both runs
        path1 = Path(run1.run_path)
        path2 = Path(run2.run_path)
        
        img1_path = path1 / 'final_result.jpg'
        img2_path = path2 / 'final_result.jpg'
        
        if not img1_path.exists() or not img2_path.exists():
            logger.warning("Final images not found, using step images")
            # Try to get last step image
            steps1 = [s for s in run1.steps if s.get('image_file')]
            steps2 = [s for s in run2.steps if s.get('image_file')]
            if steps1 and steps2:
                img1_path = path1 / steps1[-1]['image_file']
                img2_path = path2 / steps2[-1]['image_file']
        
        img1 = cv2.imread(str(img1_path)) if img1_path.exists() else None
        img2 = cv2.imread(str(img2_path)) if img2_path.exists() else None
        
        if img1 is None or img2 is None:
            raise ValueError("Could not load images for comparison")
        
        # Create side-by-side comparison
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        
        # Resize to same height
        max_h = max(h1, h2)
        if h1 != max_h:
            scale = max_h / h1
            img1 = cv2.resize(img1, (int(w1 * scale), max_h))
        if h2 != max_h:
            scale = max_h / h2
            img2 = cv2.resize(img2, (int(w2 * scale), max_h))
        
        # Combine side by side
        comparison = np.hstack([img1, img2])
        
        # Add labels
        cv2.putText(comparison, f"Run 1: {run1.image_name}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        cv2.putText(comparison, f"Run 2: {run2.image_name}", (img1.shape[1] + 10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)
        
        if output_path:
            cv2.imwrite(output_path, comparison)
        
        return comparison
    
    def generate_report(
        self,
        comparison: ComparisonResult,
        output_path: str,
        format: str = "html"
    ) -> str:
        """
        Generate comparison report.
        
        Args:
            comparison: ComparisonResult to report on
            output_path: Path to save report
            format: Report format ("html", "json")
            
        Returns:
            Path to saved report
        """
        if format == "html":
            return self._generate_html_report(comparison, output_path)
        elif format == "json":
            return self._generate_json_report(comparison, output_path)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def _compare_parameters(self, params1: Dict, params2: Dict) -> Dict[str, Any]:
        """Compare parameters between two runs."""
        all_keys = set(params1.keys()) | set(params2.keys())
        differences = {}
        
        for key in all_keys:
            val1 = params1.get(key)
            val2 = params2.get(key)
            if val1 != val2:
                differences[key] = {
                    'run1': val1,
                    'run2': val2
                }
        
        return differences
    
    def _compare_steps(self, steps1: List[Dict], steps2: List[Dict]) -> Dict[str, Any]:
        """Compare steps between two runs."""
        return {
            'count1': len(steps1),
            'count2': len(steps2),
            'step_names1': [s.get('step_name') for s in steps1],
            'step_names2': [s.get('step_name') for s in steps2]
        }
    
    def _compare_metadata(self, meta1: Dict, meta2: Dict) -> Dict[str, Any]:
        """Compare metadata between two runs."""
        differences = {}
        all_keys = set(meta1.keys()) | set(meta2.keys())
        
        for key in all_keys:
            val1 = meta1.get(key)
            val2 = meta2.get(key)
            if val1 != val2:
                differences[key] = {'run1': val1, 'run2': val2}
        
        return differences
    
    def _generate_html_report(self, comparison: ComparisonResult, output_path: str) -> str:
        """Generate HTML comparison report."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create visual diff
        visual_diff_path = str(path.parent / 'visual_comparison.jpg')
        self.visualize_differences(comparison.run1, comparison.run2, visual_diff_path)
        comparison.visual_diff_path = visual_diff_path
        
        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Comparison Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .comparison {{ margin: 20px 0; }}
        .diff {{ background: #f0f0f0; padding: 10px; margin: 5px 0; }}
        .param-diff {{ color: #d00; }}
        img {{ max-width: 100%; }}
    </style>
</head>
<body>
    <h1>Comparison Report</h1>
    
    <h2>Runs</h2>
    <p><strong>Run 1:</strong> {comparison.run1.image_name}</p>
    <p><strong>Run 2:</strong> {comparison.run2.image_name}</p>
    
    <h2>Visual Comparison</h2>
    <div class="comparison">
        <img src="{Path(visual_diff_path).name}" alt="Comparison">
    </div>
    
    <h2>Parameter Differences</h2>
    <div class="diff">
        {self._format_parameter_diffs(comparison.differences.get('parameters', {}))}
    </div>
    
    <h2>Step Differences</h2>
    <div class="diff">
        <p>Run 1: {comparison.differences['steps']['count1']} steps</p>
        <p>Run 2: {comparison.differences['steps']['count2']} steps</p>
    </div>
</body>
</html>
"""
        
        with open(path, 'w') as f:
            f.write(html)
        
        return str(path)
    
    def _generate_json_report(self, comparison: ComparisonResult, output_path: str) -> str:
        """Generate JSON comparison report."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        report = {
            'run1': {
                'path': comparison.run1.run_path,
                'image_name': comparison.run1.image_name,
                'parameters': comparison.run1.parameters
            },
            'run2': {
                'path': comparison.run2.run_path,
                'image_name': comparison.run2.image_name,
                'parameters': comparison.run2.parameters
            },
            'differences': comparison.differences
        }
        
        with open(path, 'w') as f:
            json.dump(report, f, indent=2)
        
        return str(path)
    
    def _format_parameter_diffs(self, diffs: Dict[str, Any]) -> str:
        """Format parameter differences as HTML."""
        if not diffs:
            return "<p>No parameter differences</p>"
        
        html = "<ul>"
        for key, values in diffs.items():
            html += f"<li class='param-diff'><strong>{key}:</strong> {values['run1']} → {values['run2']}</li>"
        html += "</ul>"
        return html

