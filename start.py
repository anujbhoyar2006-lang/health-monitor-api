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
    logging.info("ROOT listing: %s", os.listdir(ROOT))
    logging.info("backend exists: %s", os.path.isdir(os.path.join(ROOT, "backend")))
    logging.debug("CWD: %s", os.getcwd())
    try:
        # Import the app module directly to ensure import uses our modified sys.path
        try:
            import backend.main as main_mod
            app_obj = main_mod.app
            logging.info("Imported backend.main successfully.")
        except Exception:
            logging.exception("Failed to import backend.main. Directory listing and sys.path below for diagnostics:")
            logging.error("ROOT listing: %s", os.listdir(ROOT))
            logging.error("sys.path: %s", sys.path[:5])
            raise

        uvicorn.run(app_obj, host="0.0.0.0", port=port, reload=False)
    except Exception:
        logging.exception("Failed to start the server")
        raise
