import time
import decorator
from loguru import logger

def retry(howmany, *exception_types, **kwargs):
    timesleep = kwargs.get("timesleep", 10.0)  # seconds

    @decorator.decorator
    def tryIt(func, *fargs, **fkwargs):
        for _ in range(howmany):
            try:
                return func(*fargs, **fkwargs)
            except exception_types or Exception:
                if timesleep is not None:
                    time.sleep(timesleep)
                    logger.info(f"Retry {func.__name__}")

    return tryIt