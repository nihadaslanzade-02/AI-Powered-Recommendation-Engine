"""WSGI entry point.

The application itself lives in ``recengine.api``. This module stays at the
repository root so that ``gunicorn app:app``, which is what the Dockerfile and
every existing bookmark use, keeps working.
"""

from recengine.api import app, create_app

__all__ = ["app", "create_app"]

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000)
