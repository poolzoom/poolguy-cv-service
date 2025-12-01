#!/usr/bin/env python3
"""
Start web review server.

Provides web interface for reviewing test results and experiments.
"""

import sys
import os
import argparse
from pathlib import Path

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from systems.web_review_server import WebReviewServer
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description='Start web review server')
    parser.add_argument('--experiments-dir', type=str, default='experiments',
                       help='Directory containing experiment results (default: experiments)')
    parser.add_argument('--port', type=int, default=8080,
                       help='Port to serve on (default: 8080)')
    parser.add_argument('--no-browser', action='store_true',
                       help='Do not automatically open browser')
    
    args = parser.parse_args()
    
    # Check if experiments directory exists
    exp_dir = Path(args.experiments_dir)
    if not exp_dir.exists():
        logger.warning(f"Experiments directory does not exist: {exp_dir}")
        logger.info(f"Creating directory: {exp_dir}")
        exp_dir.mkdir(parents=True, exist_ok=True)
    
    # Start server
    server = WebReviewServer(experiments_dir=str(exp_dir), port=args.port)
    
    try:
        logger.info(f"Starting web review server...")
        logger.info(f"Experiments directory: {exp_dir}")
        logger.info(f"Server will be available at: http://localhost:{args.port}")
        
        server.start(open_browser=not args.no_browser)
        
        logger.info("Server running. Press Ctrl+C to stop.")
        
        # Keep server running
        import time
        while True:
            time.sleep(1)
    
    except KeyboardInterrupt:
        logger.info("\nStopping server...")
        server.stop()
        logger.info("Server stopped.")


if __name__ == '__main__':
    main()








