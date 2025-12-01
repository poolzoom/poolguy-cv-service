"""
WSGI entry point for PoolGuy CV Service.

This file is used by Gunicorn and other WSGI servers to run the application
in production/staging environments.

Usage:
    gunicorn -w 4 -b 0.0.0.0:5000 --timeout 120 wsgi:app
"""

from app import app

# Export app for WSGI servers
application = app

if __name__ == '__main__':
    # This allows running directly with: python wsgi.py
    # But in production, use Gunicorn instead
    import os
    port = int(os.getenv('PORT', 5000))
    debug = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'
    app.run(host='0.0.0.0', port=port, debug=debug)
