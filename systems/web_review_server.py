"""
Web Review Server System.

Serves interactive web interface for reviewing test results, experiments, and comparisons.
Can be used as standalone server or integrated into Flask app.
"""

import json
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
import urllib.parse
import webbrowser
import threading
import time

logger = logging.getLogger(__name__)


def create_review_routes(flask_app, experiments_dir: str = "experiments"):
    """
    Create Flask routes for review interface.
    
    Args:
        flask_app: Flask application instance
        experiments_dir: Directory containing experiment results
    """
    from flask import jsonify, send_from_directory
    
    exp_dir = Path(experiments_dir)
    
    @flask_app.route('/review')
    @flask_app.route('/review/')
    def review_index():
        """Serve experiment browser index page."""
        html = _load_template('index.html')
        experiments = _list_experiments(exp_dir)
        
        # Generate experiment list
        exp_list_html = ""
        for exp in experiments:
            error_indicator = ""
            if exp.get('has_error', False):
                error_indicator = '<span style="color: orange; font-weight: bold;">⚠️ Partial Data</span>'
            
            exp_list_html += f"""
            <div class="experiment-item" style="{'border-left: 3px solid orange;' if exp.get('has_error') else ''}">
                <h3><a href="/review/run/{exp['path']}">{exp['name']}</a> {error_indicator}</h3>
                <p>Image: {exp['image_name']}</p>
                <p>Steps: {exp['step_count']}</p>
                <p>Timestamp: {exp.get('timestamp', 'N/A')}</p>
                {f'<p style="color: #666; font-size: 0.9em;">Note: {exp.get("error_message", "Some data may be missing")}</p>' if exp.get('has_error') else ''}
            </div>
            """
        
        html = html.replace('{{EXPERIMENTS}}', exp_list_html)
        return html
    
    @flask_app.route('/review/run/<path:run_path>')
    def review_run_viewer(run_path: str):
        """Serve single run viewer."""
        run_dir = exp_dir / run_path
        
        if not run_dir.exists():
            return "Run not found", 404
        
        # Load log.json
        log_file = run_dir / 'log.json'
        if not log_file.exists():
            return "Log file not found", 404
        
        with open(log_file, 'r') as f:
            log_data = json.load(f)
        
        html = _load_template('run_viewer.html')
        
        # Generate step list - sort steps by step_id for logical ordering
        steps = log_data.get('steps', [])
        
        # Sort steps: handle two-part numbering (00_XX, 01_XX, 02_XX, etc.)
        def sort_key(step):
            step_name = step.get('step_name', '')
            original_idx = steps.index(step)  # Preserve original order for non-numeric steps
            import re
            # Match two-part numbering: 01_00_setup, 02_01_yolo_conf, etc.
            match = re.match(r'^(\d+)_(\d+)_(.+)$', step_name)
            if match:
                # Two-part numbering: (stage, substep, name)
                stage = int(match.group(1))
                substep = int(match.group(2))
                name = match.group(3)
                return (0, stage, substep, name)
            # Match single-part numbering: 00_strip_detection, etc.
            match = re.match(r'^(\d+)_(.+)$', step_name)
            if match:
                stage = int(match.group(1))
                name = match.group(2)
                return (0, stage, 0, name)  # Single-part gets substep 0
            # Non-numeric steps: preserve original order but place after numbered steps
            return (1, 10000 + original_idx, 0, step_name)
        
        sorted_steps = sorted(steps, key=sort_key)
        
        steps_html = ""
        for idx, step in enumerate(sorted_steps):
            original_idx = steps.index(step)  # Keep original index for image file reference
            step_file = step.get('image_file', '')
            step_name = step.get('step_name', 'Unknown')
            step_desc = step.get('description', '')
            step_data = step.get('data', {})
            
            # Create image path
            img_path = f"/review/static/{run_path}/{step_file}" if step_file else ""
            
            # Properly escape for HTML attributes
            step_name_escaped = step_name.replace("'", "&#39;").replace('"', '&quot;').replace('&', '&amp;')
            step_desc_escaped = step_desc.replace("'", "&#39;").replace('"', '&quot;').replace('&', '&amp;')
            # Escape JSON for HTML attribute (use single quotes in HTML, escape single quotes in JSON)
            step_data_json = json.dumps(step_data).replace("'", "&#39;").replace('"', '&quot;')
            
            steps_html += f"""
            <div class="step-item" 
                 data-step-idx="{original_idx}"
                 data-step-name="{step_name_escaped}"
                 data-step-image="{img_path}"
                 data-step-data="{step_data_json}"
                 onclick="showStepFromData(this)">
                <h4>{step_name}</h4>
                <p>{step_desc}</p>
                {f'<img src="{img_path}" alt="{step_name}" class="step-thumbnail">' if img_path else '<p>No image</p>'}
            </div>
            """
        
        html = html.replace('{{STEPS}}', steps_html)
        html = html.replace('{{IMAGE_NAME}}', log_data.get('image_name', 'Unknown'))
        html = html.replace('{{RUN_PATH}}', run_path)
        
        return html
    
    @flask_app.route('/review/static/<path:file_path>')
    def review_static(file_path: str):
        """Serve static files (images, CSS, JS)."""
        try:
            # Security: Prevent path traversal attacks
            import os
            normalized = os.path.normpath(file_path)
            if '..' in normalized or normalized.startswith('/'):
                logger.warning(f"Path traversal attempt detected: {file_path}")
                return "Invalid path", 400
            
            # Handle CSS/JS files from web_review/static
            if file_path.startswith('css/') or file_path.startswith('js/'):
                static_dir = Path(__file__).parent.parent / 'web_review' / 'static'
                target_path = static_dir / normalized
                
                # Ensure path is within static_dir (prevent traversal)
                try:
                    target_path.resolve().relative_to(static_dir.resolve())
                except ValueError:
                    logger.warning(f"Path outside static directory: {file_path}")
                    return "Invalid path", 403
                
                if not target_path.exists():
                    return "File not found", 404
                
                return send_from_directory(str(static_dir), normalized)
            
            # Handle image files from experiments directory
            file_path_obj = exp_dir / normalized
            
            # Ensure path is within experiments directory (prevent traversal)
            try:
                file_path_obj.resolve().relative_to(exp_dir.resolve())
            except ValueError:
                logger.warning(f"Path outside experiments directory: {file_path}")
                return "Invalid path", 403
            
            if not file_path_obj.exists():
                logger.warning(f"File not found: {file_path_obj}")
                return "File not found", 404
            
            if not file_path_obj.is_file():
                logger.warning(f"Path is not a file: {file_path_obj}")
                return "Path is not a file", 404
            
            # Extract directory and filename
            return send_from_directory(str(file_path_obj.parent), file_path_obj.name)
        except Exception as e:
            logger.error(f"Error serving static file {file_path}: {e}", exc_info=True)
            return f"Error serving file: {str(e)}", 500
    
    @flask_app.route('/review/api/experiments')
    def review_api_experiments():
        """Serve experiments list as JSON."""
        experiments = _list_experiments(exp_dir)
        return jsonify({'experiments': experiments})
    
    @flask_app.route('/review/api/run/<path:run_path>')
    def review_api_run(run_path: str):
        """Serve run data as JSON."""
        run_dir = exp_dir / run_path
        
        log_file = run_dir / 'log.json'
        if not log_file.exists():
            return "Log file not found", 404
        
        with open(log_file, 'r') as f:
            log_data = json.load(f)
        
        return jsonify(log_data)


def _list_experiments(experiments_dir: Path) -> List[Dict[str, Any]]:
    """List all experiments in experiments directory."""
    import re
    
    experiments = []
    
    # Walk through experiments directory
    for exp_dir in experiments_dir.rglob('*'):
        if exp_dir.is_dir() and (exp_dir / 'log.json').exists():
            log_file = exp_dir / 'log.json'
            has_error = False
            
            # Try full JSON parse first
            try:
                with open(log_file, 'r') as f:
                    log_data = json.load(f)
                
                experiments.append({
                    'name': exp_dir.name,
                    'path': str(exp_dir.relative_to(experiments_dir)),
                    'image_name': log_data.get('image_name', 'Unknown'),
                    'step_count': len(log_data.get('steps', [])),
                    'timestamp': log_data.get('timestamp', ''),
                    'has_error': False
                })
            except (json.JSONDecodeError, ValueError) as e:
                # JSON is corrupted - try to extract basic info from file content
                has_error = True
                logger.warning(f"Failed to parse JSON for {exp_dir}: {e}")
                
                try:
                    with open(log_file, 'r') as f:
                        content = f.read()
                    
                    # Try to extract image_name and timestamp using regex
                    image_name_match = re.search(r'"image_name"\s*:\s*"([^"]+)"', content)
                    timestamp_match = re.search(r'"timestamp"\s*:\s*"([^"]+)"', content)
                    step_count_match = re.findall(r'"step_name"', content)
                    
                    image_name = image_name_match.group(1) if image_name_match else exp_dir.parent.name if exp_dir.parent.name != 'pipeline' else exp_dir.parent.parent.name
                    timestamp = timestamp_match.group(1) if timestamp_match else ''
                    step_count = len(step_count_match) if step_count_match else 0
                    
                    # Use directory structure to infer image name if not found
                    if image_name == 'Unknown' or not image_name:
                        # Try to get from parent directory names
                        parts = exp_dir.parts
                        for part in reversed(parts):
                            if part and part != 'pipeline' and part != 'test_single' and part != 'refinement' and part != 'experiments':
                                image_name = part
                                break
                    
                    experiments.append({
                        'name': exp_dir.name,
                        'path': str(exp_dir.relative_to(experiments_dir)),
                        'image_name': image_name,
                        'step_count': step_count,
                        'timestamp': timestamp,
                        'has_error': True,
                        'error_message': f'Corrupted JSON: {str(e)[:50]}'
                    })
                except Exception as e2:
                    # Even fallback parsing failed - use directory structure
                    logger.warning(f"Fallback parsing also failed for {exp_dir}: {e2}")
                    parts = exp_dir.parts
                    image_name = exp_dir.name
                    for part in reversed(parts):
                        if part and part != 'pipeline' and part != 'test_single' and part != 'refinement' and part != 'experiments':
                            image_name = part
                            break
                    
                    experiments.append({
                        'name': exp_dir.name,
                        'path': str(exp_dir.relative_to(experiments_dir)),
                        'image_name': image_name,
                        'step_count': 0,
                        'timestamp': '',
                        'has_error': True,
                        'error_message': 'Unable to parse log file'
                    })
            except Exception as e:
                # Any other error - still include the experiment
                logger.warning(f"Unexpected error loading experiment {exp_dir}: {e}")
                experiments.append({
                    'name': exp_dir.name,
                    'path': str(exp_dir.relative_to(experiments_dir)),
                    'image_name': exp_dir.name,
                    'step_count': 0,
                    'timestamp': '',
                    'has_error': True,
                    'error_message': str(e)[:50]
                })
    
    return sorted(experiments, key=lambda x: x.get('timestamp', ''), reverse=True)


def _load_template(template_name: str) -> str:
    """Load HTML template."""
    template_path = Path(__file__).parent.parent / 'web_review' / 'templates' / template_name
    
    if template_path.exists():
        with open(template_path, 'r') as f:
            return f.read()
    else:
        # Return basic template if file doesn't exist
        return _get_default_template(template_name)


def _get_default_template(template_name: str) -> str:
    """Get default template HTML."""
    if template_name == 'index.html':
        return """<!DOCTYPE html>
<html>
<head>
    <title>Experiment Browser</title>
    <link rel="stylesheet" href="/review/static/css/review.css">
</head>
<body>
    <div class="container">
        <h1>Experiment Browser</h1>
        {{EXPERIMENTS}}
    </div>
</body>
</html>
"""
    elif template_name == 'run_viewer.html':
        return """<!DOCTYPE html>
<html>
<head>
    <title>Run Viewer - {{IMAGE_NAME}}</title>
    <link rel="stylesheet" href="/review/static/css/review.css">
</head>
<body>
    <div class="container">
        <h1>Run Viewer: {{IMAGE_NAME}}</h1>
        <a href="/review">← Back to Experiments</a>
        <div id="steps" class="steps-container">{{STEPS}}</div>
        <div id="step-detail" class="step-detail" style="display: none;">
            <h3 id="step-detail-name"></h3>
            <img id="step-detail-image" src="" alt="Step detail" class="step-detail-image">
            <div id="step-detail-data" class="step-detail-data"></div>
        </div>
    </div>
    <script>
        function showStep(idx, stepName, imagePath, data) {
            const detailDiv = document.getElementById('step-detail');
            document.getElementById('step-detail-name').textContent = stepName;
            if (imagePath) {
                document.getElementById('step-detail-image').src = imagePath;
                document.getElementById('step-detail-image').style.display = 'block';
            } else {
                document.getElementById('step-detail-image').style.display = 'none';
            }
            document.getElementById('step-detail-data').innerHTML = '<pre>' + JSON.stringify(data, null, 2) + '</pre>';
            detailDiv.style.display = 'block';
            detailDiv.scrollIntoView({ behavior: 'smooth' });
        }
    </script>
</body>
</html>
"""
    else:
        return "<html><body><h1>Template not found</h1></body></html>"


def create_handler(experiments_dir: str):
    """Factory function to create request handler with experiments_dir."""
    class ReviewRequestHandler(BaseHTTPRequestHandler):
        """HTTP request handler for review server."""
        
        def __init__(self, *args, **kwargs):
            self.experiments_dir = Path(experiments_dir)
            super().__init__(*args, **kwargs)
        
        def do_GET(self):
            """Handle GET requests."""
            parsed_path = urllib.parse.urlparse(self.path)
            path = parsed_path.path
            
            if path == '/' or path == '/index.html':
                self._serve_index()
            elif path.startswith('/run/'):
                self._serve_run_viewer(path)
            elif path.startswith('/compare/'):
                self._serve_comparison(path)
            elif path.startswith('/api/experiments'):
                self._serve_api_experiments()
            elif path.startswith('/api/run/'):
                self._serve_api_run(path)
            elif path.startswith('/static/'):
                self._serve_static(path)
            else:
                self._serve_404()
        
        def _serve_index(self):
            """Serve experiment browser index page."""
            html = self._load_template('index.html')
            experiments = self._list_experiments()
            
            # Generate experiment list
            exp_list_html = ""
            for exp in experiments:
                error_indicator = ""
                if exp.get('has_error', False):
                    error_indicator = '<span style="color: orange; font-weight: bold;">⚠️ Partial Data</span>'
                
                exp_list_html += f"""
            <div class="experiment-item" style="{'border-left: 3px solid orange;' if exp.get('has_error') else ''}">
                <h3><a href="/run/{exp['path']}">{exp['name']}</a> {error_indicator}</h3>
                <p>Image: {exp['image_name']}</p>
                <p>Steps: {exp['step_count']}</p>
                <p>Timestamp: {exp.get('timestamp', 'N/A')}</p>
                {f'<p style="color: #666; font-size: 0.9em;">Note: {exp.get("error_message", "Some data may be missing")}</p>' if exp.get('has_error') else ''}
            </div>
            """
            
            html = html.replace('{{EXPERIMENTS}}', exp_list_html)
            
            self._send_response(200, html, 'text/html')
        
        def _serve_run_viewer(self, path: str):
            """Serve single run viewer."""
            # Extract run path from URL
            run_path = urllib.parse.unquote(path.replace('/run/', ''))
            run_dir = self.experiments_dir / run_path
            
            if not run_dir.exists():
                self._serve_404()
                return
            
            # Load log.json
            log_file = run_dir / 'log.json'
            if not log_file.exists():
                self._serve_404()
                return
            
            with open(log_file, 'r') as f:
                log_data = json.load(f)
            
            html = self._load_template('run_viewer.html')
            
            # Generate step list
            steps_html = ""
            for idx, step in enumerate(log_data.get('steps', [])):
                step_file = step.get('image_file', '')
                step_name = step.get('step_name', 'Unknown')
                step_desc = step.get('description', '')
                step_data = step.get('data', {})
                
                # Create image path
                img_path = f"/static/{run_path}/{step_file}" if step_file else ""
                
                # Escape step_name and step_desc for HTML
                step_name_escaped = step_name.replace("'", "\\'")
                step_desc_escaped = step_desc.replace("'", "\\'")
                step_data_json = json.dumps(step_data).replace("'", "\\'")
                
                steps_html += f"""
                <div class="step-item" onclick="showStep({idx}, '{step_name_escaped}', '{img_path}', {step_data_json})">
                    <h4>{step_name}</h4>
                    <p>{step_desc}</p>
                    {f'<img src="{img_path}" alt="{step_name}" class="step-thumbnail">' if img_path else '<p>No image</p>'}
                </div>
                """
            
            html = html.replace('{{STEPS}}', steps_html)
            html = html.replace('{{IMAGE_NAME}}', log_data.get('image_name', 'Unknown'))
            html = html.replace('{{RUN_PATH}}', run_path)
            
            self._send_response(200, html, 'text/html')
        
        def _serve_comparison(self, path: str):
            """Serve comparison viewer."""
            # Extract run paths from URL
            parts = path.replace('/compare/', '').split('/')
            if len(parts) < 2:
                self._serve_404()
                return
            
            run1_path = urllib.parse.unquote(parts[0])
            run2_path = urllib.parse.unquote(parts[1])
            
            html = self._load_template('comparison.html')
            html = html.replace('{{RUN1_PATH}}', run1_path)
            html = html.replace('{{RUN2_PATH}}', run2_path)
            
            self._send_response(200, html, 'text/html')
        
        def _serve_api_experiments(self):
            """Serve experiments list as JSON."""
            experiments = self._list_experiments()
            self._send_json_response(200, {'experiments': experiments})
        
        def _serve_api_run(self, path: str):
            """Serve run data as JSON."""
            run_path = urllib.parse.unquote(path.replace('/api/run/', ''))
            run_dir = self.experiments_dir / run_path
            
            log_file = run_dir / 'log.json'
            if not log_file.exists():
                self._serve_404()
                return
            
            with open(log_file, 'r') as f:
                log_data = json.load(f)
            
            self._send_json_response(200, log_data)
        
        def _serve_static(self, path: str):
            """Serve static files (images, CSS, JS)."""
            file_path = self.experiments_dir / path.replace('/static/', '')
            
            if not file_path.exists():
                self._serve_404()
                return
            
            # Determine content type
            content_type = 'application/octet-stream'
            if file_path.suffix == '.jpg' or file_path.suffix == '.jpeg':
                content_type = 'image/jpeg'
            elif file_path.suffix == '.png':
                content_type = 'image/png'
            elif file_path.suffix == '.css':
                content_type = 'text/css'
            elif file_path.suffix == '.js':
                content_type = 'application/javascript'
            
            with open(file_path, 'rb') as f:
                content = f.read()
            
            self._send_response(200, content, content_type, binary=True)
        
        def _list_experiments(self) -> List[Dict[str, Any]]:
            """List all experiments in experiments directory."""
            experiments = []
            
            # Walk through experiments directory
            for exp_dir in self.experiments_dir.rglob('*'):
                if exp_dir.is_dir() and (exp_dir / 'log.json').exists():
                    try:
                        with open(exp_dir / 'log.json', 'r') as f:
                            log_data = json.load(f)
                        
                        experiments.append({
                            'name': exp_dir.name,
                            'path': str(exp_dir.relative_to(self.experiments_dir)),
                            'image_name': log_data.get('image_name', 'Unknown'),
                            'step_count': len(log_data.get('steps', [])),
                            'timestamp': log_data.get('timestamp', '')
                        })
                    except Exception as e:
                        logger.warning(f"Failed to load experiment {exp_dir}: {e}")
            
            return sorted(experiments, key=lambda x: x.get('timestamp', ''), reverse=True)
        
        def _load_template(self, template_name: str) -> str:
            """Load HTML template."""
            template_path = Path(__file__).parent.parent / 'web_review' / 'templates' / template_name
            
            if template_path.exists():
                with open(template_path, 'r') as f:
                    return f.read()
            else:
                # Return basic template if file doesn't exist
                return self._get_default_template(template_name)
        
        def _get_default_template(self, template_name: str) -> str:
            """Get default template HTML."""
            if template_name == 'index.html':
                return """<!DOCTYPE html>
<html>
<head>
    <title>Experiment Browser</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .experiment-item { border: 1px solid #ddd; padding: 10px; margin: 10px 0; }
    </style>
</head>
<body>
    <h1>Experiment Browser</h1>
    {{EXPERIMENTS}}
</body>
</html>
"""
            elif template_name == 'run_viewer.html':
                return """<!DOCTYPE html>
<html>
<head>
    <title>Run Viewer - {{IMAGE_NAME}}</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .step-item { border: 1px solid #ddd; padding: 10px; margin: 10px 0; }
        .step-thumbnail { max-width: 200px; cursor: pointer; }
        .step-data { display: none; }
        .step-data.active { display: block; }
    </style>
</head>
<body>
    <h1>Run Viewer: {{IMAGE_NAME}}</h1>
    <div id="steps">{{STEPS}}</div>
    <script>
        function showStep(idx) {
            const items = document.querySelectorAll('.step-data');
            items.forEach(item => item.classList.remove('active'));
            items[idx].classList.add('active');
        }
    </script>
</body>
</html>
"""
            else:
                return "<html><body><h1>Template not found</h1></body></html>"
        
        def _send_response(self, status: int, content: Any, content_type: str, binary: bool = False):
            """Send HTTP response."""
            self.send_response(status)
            self.send_header('Content-Type', content_type)
            self.end_headers()
            
            if binary:
                self.wfile.write(content)
            else:
                self.wfile.write(content.encode('utf-8'))
        
        def _send_json_response(self, status: int, data: Dict[str, Any]):
            """Send JSON response."""
            json_str = json.dumps(data, indent=2)
            self._send_response(status, json_str, 'application/json')
        
        def _serve_404(self):
            """Serve 404 error."""
            self._send_response(404, "<html><body><h1>404 Not Found</h1></body></html>", 'text/html')
        
        def log_message(self, format, *args):
            """Override to use our logger."""
            logger.info(f"{self.address_string()} - {format % args}")
    
    return ReviewRequestHandler
    
    def do_GET(self):
        """Handle GET requests."""
        parsed_path = urllib.parse.urlparse(self.path)
        path = parsed_path.path
        
        if path == '/' or path == '/index.html':
            self._serve_index()
        elif path.startswith('/run/'):
            self._serve_run_viewer(path)
        elif path.startswith('/compare/'):
            self._serve_comparison(path)
        elif path.startswith('/api/experiments'):
            self._serve_api_experiments()
        elif path.startswith('/api/run/'):
            self._serve_api_run(path)
        elif path.startswith('/static/'):
            self._serve_static(path)
        else:
            self._serve_404()
    
    def _serve_index(self):
        """Serve experiment browser index page."""
        html = self._load_template('index.html')
        experiments = self._list_experiments()
        
        # Generate experiment list
        exp_list_html = ""
        for exp in experiments:
                error_indicator = ""
                if exp.get('has_error', False):
                    error_indicator = '<span style="color: orange; font-weight: bold;">⚠️ Partial Data</span>'
                
                exp_list_html += f"""
            <div class="experiment-item" style="{'border-left: 3px solid orange;' if exp.get('has_error') else ''}">
                <h3><a href="/run/{exp['path']}">{exp['name']}</a> {error_indicator}</h3>
                <p>Image: {exp['image_name']}</p>
                <p>Steps: {exp['step_count']}</p>
                <p>Timestamp: {exp.get('timestamp', 'N/A')}</p>
                {f'<p style="color: #666; font-size: 0.9em;">Note: {exp.get("error_message", "Some data may be missing")}</p>' if exp.get('has_error') else ''}
            </div>
            """
        
        html = html.replace('{{EXPERIMENTS}}', exp_list_html)
        
        self._send_response(200, html, 'text/html')
    
    def _serve_run_viewer(self, path: str):
        """Serve single run viewer."""
        # Extract run path from URL
        run_path = urllib.parse.unquote(path.replace('/run/', ''))
        run_dir = self.experiments_dir / run_path
        
        if not run_dir.exists():
            self._serve_404()
            return
        
        # Load log.json
        log_file = run_dir / 'log.json'
        if not log_file.exists():
            self._serve_404()
            return
        
        with open(log_file, 'r') as f:
            log_data = json.load(f)
        
        html = self._load_template('run_viewer.html')
        
        # Generate step list - sort steps by step_id for logical ordering
        steps = log_data.get('steps', [])
        
        # Sort steps: handle two-part numbering (00_XX, 01_XX, 02_XX, etc.)
        def sort_key(step):
            step_name = step.get('step_name', '')
            original_idx = steps.index(step)  # Preserve original order for non-numeric steps
            import re
            # Match two-part numbering: 01_00_setup, 02_01_yolo_conf, etc.
            match = re.match(r'^(\d+)_(\d+)_(.+)$', step_name)
            if match:
                # Two-part numbering: (stage, substep, name)
                stage = int(match.group(1))
                substep = int(match.group(2))
                name = match.group(3)
                return (0, stage, substep, name)
            # Match single-part numbering: 00_strip_detection, etc.
            match = re.match(r'^(\d+)_(.+)$', step_name)
            if match:
                stage = int(match.group(1))
                name = match.group(2)
                return (0, stage, 0, name)  # Single-part gets substep 0
            # Non-numeric steps: preserve original order but place after numbered steps
            return (1, 10000 + original_idx, 0, step_name)
        
        sorted_steps = sorted(steps, key=sort_key)
        
        steps_html = ""
        for idx, step in enumerate(sorted_steps):
            original_idx = steps.index(step)  # Keep original index for image file reference
            step_file = step.get('image_file', '')
            steps_html += f"""
            <div class="step-item" data-step="{original_idx}">
                <h4>{step.get('step_name', 'Unknown')}</h4>
                <p>{step.get('description', '')}</p>
                <img src="/static/{run_path}/{step_file}" alt="{step.get('step_name')}" 
                     onclick="showStep({original_idx})" class="step-thumbnail">
                <div class="step-data">
                    <pre>{json.dumps(step.get('data', {}), indent=2)}</pre>
                </div>
            </div>
            """
        
        html = html.replace('{{STEPS}}', steps_html)
        html = html.replace('{{IMAGE_NAME}}', log_data.get('image_name', 'Unknown'))
        html = html.replace('{{RUN_PATH}}', run_path)
        
        self._send_response(200, html, 'text/html')
    
    def _serve_comparison(self, path: str):
        """Serve comparison viewer."""
        # Extract run paths from URL
        parts = path.replace('/compare/', '').split('/')
        if len(parts) < 2:
            self._serve_404()
            return
        
        run1_path = urllib.parse.unquote(parts[0])
        run2_path = urllib.parse.unquote(parts[1])
        
        html = self._load_template('comparison.html')
        html = html.replace('{{RUN1_PATH}}', run1_path)
        html = html.replace('{{RUN2_PATH}}', run2_path)
        
        self._send_response(200, html, 'text/html')
    
    def _serve_api_experiments(self):
        """Serve experiments list as JSON."""
        experiments = self._list_experiments()
        self._send_json_response(200, {'experiments': experiments})
    
    def _serve_api_run(self, path: str):
        """Serve run data as JSON."""
        run_path = urllib.parse.unquote(path.replace('/api/run/', ''))
        run_dir = self.experiments_dir / run_path
        
        log_file = run_dir / 'log.json'
        if not log_file.exists():
            self._serve_404()
            return
        
        with open(log_file, 'r') as f:
            log_data = json.load(f)
        
        self._send_json_response(200, log_data)
    
    def _serve_static(self, path: str):
        """Serve static files (images, CSS, JS)."""
        file_path = self.experiments_dir / path.replace('/static/', '')
        
        if not file_path.exists():
            self._serve_404()
            return
        
        # Determine content type
        content_type = 'application/octet-stream'
        if file_path.suffix == '.jpg' or file_path.suffix == '.jpeg':
            content_type = 'image/jpeg'
        elif file_path.suffix == '.png':
            content_type = 'image/png'
        elif file_path.suffix == '.css':
            content_type = 'text/css'
        elif file_path.suffix == '.js':
            content_type = 'application/javascript'
        
        with open(file_path, 'rb') as f:
            content = f.read()
        
        self._send_response(200, content, content_type, binary=True)
    
    def _list_experiments(self) -> List[Dict[str, Any]]:
        """List all experiments in experiments directory."""
        experiments = []
        
        # Walk through experiments directory
        for exp_dir in self.experiments_dir.rglob('*'):
            if exp_dir.is_dir() and (exp_dir / 'log.json').exists():
                try:
                    with open(exp_dir / 'log.json', 'r') as f:
                        log_data = json.load(f)
                    
                    experiments.append({
                        'name': exp_dir.name,
                        'path': str(exp_dir.relative_to(self.experiments_dir)),
                        'image_name': log_data.get('image_name', 'Unknown'),
                        'step_count': len(log_data.get('steps', [])),
                        'timestamp': log_data.get('timestamp', '')
                    })
                except Exception as e:
                    logger.warning(f"Failed to load experiment {exp_dir}: {e}")
        
        return sorted(experiments, key=lambda x: x.get('timestamp', ''), reverse=True)
    
    def _load_template(self, template_name: str) -> str:
        """Load HTML template."""
        template_path = Path(__file__).parent.parent / 'web_review' / 'templates' / template_name
        
        if template_path.exists():
            with open(template_path, 'r') as f:
                return f.read()
        else:
            # Return basic template if file doesn't exist
            return self._get_default_template(template_name)
    
    def _get_default_template(self, template_name: str) -> str:
        """Get default template HTML."""
        if template_name == 'index.html':
            return """<!DOCTYPE html>
<html>
<head>
    <title>Experiment Browser</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .experiment-item { border: 1px solid #ddd; padding: 10px; margin: 10px 0; }
    </style>
</head>
<body>
    <h1>Experiment Browser</h1>
    {{EXPERIMENTS}}
</body>
</html>
"""
        elif template_name == 'run_viewer.html':
            return """<!DOCTYPE html>
<html>
<head>
    <title>Run Viewer - {{IMAGE_NAME}}</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .step-item { border: 1px solid #ddd; padding: 10px; margin: 10px 0; }
        .step-thumbnail { max-width: 200px; cursor: pointer; }
        .step-data { display: none; }
        .step-data.active { display: block; }
    </style>
</head>
<body>
    <h1>Run Viewer: {{IMAGE_NAME}}</h1>
    <div id="steps">{{STEPS}}</div>
    <script>
        function showStep(idx) {
            const items = document.querySelectorAll('.step-data');
            items.forEach(item => item.classList.remove('active'));
            items[idx].classList.add('active');
        }
    </script>
</body>
</html>
"""
        else:
            return "<html><body><h1>Template not found</h1></body></html>"
    
    def _send_response(self, status: int, content: Any, content_type: str, binary: bool = False):
        """Send HTTP response."""
        self.send_response(status)
        self.send_header('Content-Type', content_type)
        self.end_headers()
        
        if binary:
            self.wfile.write(content)
        else:
            self.wfile.write(content.encode('utf-8'))
    
    def _send_json_response(self, status: int, data: Dict[str, Any]):
        """Send JSON response."""
        json_str = json.dumps(data, indent=2)
        self._send_response(status, json_str, 'application/json')
    
    def _serve_404(self):
        """Serve 404 error."""
        self._send_response(404, "<html><body><h1>404 Not Found</h1></body></html>", 'text/html')
    
    def log_message(self, format, *args):
        """Override to use our logger."""
        logger.info(f"{self.address_string()} - {format % args}")


class WebReviewServer:
    """
    Web server for reviewing test results and experiments.
    
    Provides interactive web interface to view debug logs, compare runs,
    and inspect step-by-step results with images.
    """
    
    def __init__(self, experiments_dir: str = "experiments", port: int = 8080):
        """
        Initialize web review server.
        
        Args:
            experiments_dir: Directory containing experiment results
            port: Port to serve on
        """
        self.experiments_dir = Path(experiments_dir)
        self.port = port
        self.server: Optional[HTTPServer] = None
        self.thread: Optional[threading.Thread] = None
    
    def start(self, open_browser: bool = True):
        """
        Start the web server.
        
        Args:
            open_browser: Whether to automatically open browser
        """
        handler_class = create_handler(str(self.experiments_dir))
        self.server = HTTPServer(('localhost', self.port), handler_class)
        
        def run_server():
            logger.info(f"Starting web review server on http://localhost:{self.port}")
            self.server.serve_forever()
        
        self.thread = threading.Thread(target=run_server, daemon=True)
        self.thread.start()
        
        # Wait a moment for server to start
        time.sleep(0.5)
        
        if open_browser:
            url = f"http://localhost:{self.port}"
            logger.info(f"Opening browser: {url}")
            webbrowser.open(url)
    
    def stop(self):
        """Stop the web server."""
        if self.server:
            self.server.shutdown()
            self.server.server_close()
            logger.info("Web review server stopped")
    
    def serve_experiment(self, experiment_path: str):
        """
        Serve a specific experiment.
        
        Args:
            experiment_path: Path to experiment directory
        """
        # This would open the specific experiment in browser
        url = f"http://localhost:{self.port}/run/{experiment_path}"
        webbrowser.open(url)
    
    def serve_comparison(self, run1_path: str, run2_path: str):
        """
        Serve comparison view for two runs.
        
        Args:
            run1_path: Path to first run
            run2_path: Path to second run
        """
        url = f"http://localhost:{self.port}/compare/{run1_path}/{run2_path}"
        webbrowser.open(url)

