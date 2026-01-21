#!/usr/bin/env python
"""
Application entry point.

Run the Flask development server:
    python run.py

For production, use a WSGI server like gunicorn:
    gunicorn "app:create_app()"
"""

from app import create_app

app = create_app()

if __name__ == "__main__":
    app.run(debug=True)
