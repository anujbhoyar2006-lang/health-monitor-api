import os
import logging
import uvicorn

# Start the FastAPI app programmatically to avoid Click/CLI issues
if __name__ == "__main__":
    logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
    port = int(os.environ.get("PORT", 8000))
    logging.info("Starting uvicorn programmatically on port %s", port)
    try:
        uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
    except Exception:
        logging.exception("Failed to start the server")
        raise
