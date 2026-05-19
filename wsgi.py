# wsgi.py — WSGI entry point for Gunicorn
#
# Gunicorn is started with: gunicorn wsgi:app
# Do not add debug=True or run the dev server here.

from app import app

if __name__ == '__main__':
    app.run()