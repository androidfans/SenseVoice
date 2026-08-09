import threading
import time


class LazyResource:
    """Load a resource on demand and release it after inactivity."""

    def __init__(self, factory, cleanup, idle_timeout_seconds, clock=None):
        self._factory = factory
        self._cleanup = cleanup
        self._idle_timeout_seconds = float(idle_timeout_seconds)
        self._clock = clock or time.monotonic
        self._lock = threading.Lock()
        self._resource = None
        self._last_activity = self._clock()

    def touch(self):
        with self._lock:
            self._last_activity = self._clock()

    def get(self):
        with self._lock:
            self._last_activity = self._clock()
            if self._resource is None:
                self._resource = self._factory()
            return self._resource

    def unload_if_idle(self):
        with self._lock:
            if self._resource is None:
                return False
            if self._clock() - self._last_activity <= self._idle_timeout_seconds:
                return False

            resource = self._resource
            self._resource = None

        # Cleanup may wait for a child process; lifecycle state is already detached.
        self._cleanup(resource)
        return True

    @property
    def is_loaded(self):
        with self._lock:
            return self._resource is not None
