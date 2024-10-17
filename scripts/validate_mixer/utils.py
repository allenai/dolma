import sys
import signal
import re
from typing import List, Set
import builtins

def keyboard_interrupt_handler(signal: int, frame: object) -> None:
    """
    Handle keyboard interrupt (Ctrl+C) gracefully.
    
    :param signal: Signal number
    :param frame: Current stack frame
    """
    print("\n\nScript interrupted by user")
    sys.exit(0)


VERBOSE = False

def set_verbose(verbose):
    global VERBOSE
    VERBOSE = verbose

def vprint(*args, **kwargs):
    if VERBOSE:
        builtins.print(*args, **kwargs)