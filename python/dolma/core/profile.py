import cProfile
import io
import pstats
from contextlib import ExitStack, contextmanager
from typing import Generator, Optional

import smart_open

from .loggers import get_logger


@contextmanager
def profiler(
    output: Optional[str] = None, sort_key: str = "tottime", lines: int = 100, human_readable: bool = True
) -> Generator[None, None, None]:
    logger = get_logger("profiler", "info")

    profile = cProfile.Profile()
    logger.info("Starting profiler...")
    profile.enable()
    yield
    profile.disable()
    logger.info("Profiler stopped.")

    if not human_readable and output is not None:
        logger.info("Dumping profiler stats in binary format to %s...", output)
        profile.dump_stats(output)
        return

    with ExitStack() as stack:
        logger.info("Printing profiler stats %s...", f"to {output}" if output is not None else "to stdout")
        output_stream = io.StringIO() if output is None else stack.enter_context(smart_open.open(output, "w"))
        ps = pstats.Stats(profile, stream=output_stream).sort_stats(sort_key)
        ps.print_stats(lines)

    logger.info("Done printing profiler stats.")
