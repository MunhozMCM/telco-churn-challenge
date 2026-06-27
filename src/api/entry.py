"""Entry point so the API can be launched as a console script / make target."""

from __future__ import annotations

import uvicorn


def run() -> None:
    """Run the API with uvicorn on localhost:8000."""
    uvicorn.run("src.api.app:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    run()
