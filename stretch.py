"""Production length correction with Google Speedy nonlinear speech scaling."""
from speedy import ENGINE,VERSION,BINARY,stretch,ready


def close_worker():
    """Speedy subprocesses finish per call; there is no resident model to close."""
