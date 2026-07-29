def version() -> str:
    """Returns the version of the compiled extension."""

def run_demo() -> None:
    """Opens the viewer window, blocking until it is closed or escape is pressed.

    Must be called from the main thread. Only one window can be opened per
    process; calling this a second time raises ``RuntimeError``.
    """
