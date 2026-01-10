import os
import sys
import logging
import uvicorn

# Ensure repository root is on PYTHONPATH
ROOT = os.path.abspath(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Change cwd to repo root so relative paths work consistently
os.chdir(ROOT)

if __name__ == "__main__":
    logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
    port = int(os.environ.get("PORT", 8000))
    logging.info("Starting uvicorn programmatically on port %s", port)
    logging.debug("CWD: %s", os.getcwd())
    try:
        uvicorn.run("backend.main:app", host="0.0.0.0", port=port, reload=False)
    except Exception:
        logging.exception("Failed to start the server")
        raise
