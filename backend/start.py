import os
import sys
import logging
import uvicorn

# Ensure repository root and backend package are on PYTHONPATH
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Change cwd to repo root so relative paths work consistently
os.chdir(ROOT)

# Start the FastAPI app programmatically to avoid Click/CLI issues
if __name__ == "__main__":
    logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
    port = int(os.environ.get("PORT", 8000))
    logging.info("Starting uvicorn programmatically on port %s", port)
    # diagnostics
    logging.debug("CWD: %s", os.getcwd())
    logging.debug("ROOT: %s", ROOT)
    logging.debug("Contains backend package: %s", os.path.isdir(os.path.join(ROOT, "backend")))
    logging.debug("sys.path[0]: %s", sys.path[0])

    try:
        # Import target using package.module so imports inside work correctly
        uvicorn.run("backend.main:app", host="0.0.0.0", port=port, reload=False)
    except Exception:
        logging.exception("Failed to start the server")
        raise
