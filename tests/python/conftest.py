import multiprocessing

import pytest


# The default multiprocessing start method is "fork" which is not compatible with
# with runtime assertions that it is set to "spawn". When running unit tests, it's
# possible to call an external library that sets the start method to the default.
# Here we set the start method to be "spawn" for all tests before executing.
@pytest.fixture(scope="session", autouse=True)
def initialize_multiprocessing_start_method():
    try:
        multiprocessing.set_start_method("spawn")
    except Exception:
        pass
