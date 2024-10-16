import sys
import signal
import re
from typing import List, Set

def keyboard_interrupt_handler(signal: int, frame: object) -> None:
    """
    Handle keyboard interrupt (Ctrl+C) gracefully.
    
    :param signal: Signal number
    :param frame: Current stack frame
    """
    print("\n\nScript interrupted by user")
    sys.exit(0)