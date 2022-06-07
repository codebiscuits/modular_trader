import time

class TimerError(Exception):
    """A custom exception used to report errors in use of Timer class"""

class Timer:
    timers = {}

    def __init__(self, name=None):
        self._start_time = None
        self.name = name

        # Add new named timers to dictionary of timers
        if name:
            self.timers.setdefault(name, 0)

    def start(self) -> None:
        """Start a new timer"""
        if self._start_time is not None:
            raise TimerError("Timer is running. Use .stop() to stop it")

        self._start_time = time.perf_counter()

    def stop(self):
        """Stop the timer, and report the elapsed time"""
        if self._start_time is None:
            raise TimerError(f"Timer is not running. Use .start() to start it")

        elapsed_time = time.perf_counter() - self._start_time
        self._start_time = None

        if elapsed_time > 5:
            print(f'{round(elapsed_time)} for {self.name}')
        if self.name:
            self.timers[self.name] += elapsed_time

        return elapsed_time