# Python 3.8 went end of life in October 2024, and it is also why the original
# image stopped building: there is no scikit-surprise wheel for it, so pip fell
# back to compiling from source and Cython failed on co_clustering.pyx.
FROM python:3.12-slim

# Never write .pyc files or buffer stdout; both only make container logs worse.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Dependencies first, so that editing source code does not invalidate the
# layer that installs them. The original copied everything before installing,
# which meant every change to app.py reinstalled the whole dependency tree.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# No build-essential or gcc. Every pinned dependency ships a manylinux wheel,
# and dropping the toolchain takes several hundred megabytes off the image.
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install --no-cache-dir --no-deps -e .

COPY data.csv ./
COPY scripts ./scripts
COPY app.py ./

# Fit the model into the image. It takes about a second and means a container
# is ready to serve the moment it starts, rather than fitting per worker on
# first request the way the original did.
RUN python scripts/train.py

# Run as an unprivileged user.
RUN useradd --create-home --uid 10001 recengine && chown -R recengine:recengine /app
USER recengine

EXPOSE 5000

# Uses the stdlib rather than adding curl just to answer this.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:5000/health').read()"

# Two workers, each loading the same prebuilt artifact. Under the original this
# would have meant two independently fitted and disagreeing models.
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "2", "--timeout", "60", \
     "--access-logfile", "-", "app:app"]
