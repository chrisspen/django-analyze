"""
Function timeout utilities taken from
http://stackoverflow.com/questions/2281850/timeout-function-if-it-takes-too-long-to-finish
"""
from functools import wraps
import errno
import os
import signal

class TimeoutError(Exception):
    pass

def timeout(seconds=10, error_message=os.strerror(errno.ETIME)):
    """
    Allows to interrupt a function call by a specific timeout threshold.
    To be used in a function decorator:
    
        @timeout
        def long_running_function1():
            ...
    
    Note, this is NOT thread-safe.
    """
    def decorator(func):
        def _handle_timeout(signum, frame):
            raise TimeoutError(error_message)

        def wrapper(*args, **kwargs):
            signal.signal(signal.SIGALRM, _handle_timeout)
            signal.alarm(seconds)
            try:
                result = func(*args, **kwargs)
            finally:
                # Cancel alarm.
                signal.alarm(0)
            return result

        return wraps(func)(wrapper)

    return decorator

class run_until_timeout(object):
    """
    Allows to interrupt a function call by a specific timeout threshold.
    To be used in a with statement:
    
        with run_until_timeout(seconds=3):
            sleep(4)
            
    """
    def __init__(self, seconds=1, error_message='Timeout'):
        self.seconds = seconds
        self.error_message = error_message
    def handle_timeout(self, signum, frame):
        raise TimeoutError(self.error_message)
    def __enter__(self):
        if self.seconds > 0:
            signal.signal(signal.SIGALRM, self.handle_timeout)
            signal.alarm(self.seconds)
    def __exit__(self, type, value, traceback):
        # Cancel alarm.
        if self.seconds > 0:
            signal.alarm(0)
        