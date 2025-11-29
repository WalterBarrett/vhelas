import threading
import time
from functools import wraps
from typing import Any

from config import get_config

cooldown_table = {}


class CooldownPool:
    def __init__(self):
        self.config = get_config()
        self._cooldown = self.config.get_setting("CooldownDefault", 1.0)
        self._last_call = 0.0
        self.lock = threading.Lock()

    def wait(self):
        now = time.time()
        elapsed = now - self._last_call
        if elapsed < self._cooldown:
            time.sleep(self._cooldown - elapsed)

    def enforce(self, success=True):
        if success:
            self._cooldown = max(self.config.get_setting("CooldownDefault", 1.0), self._cooldown * self.config.get_setting("CooldownRecoveryMultiplier", 0.9))
        else:
            self._cooldown = min(self.config.get_setting("CooldownMax", 16.0), self._cooldown + min(self._cooldown, 1.0))

        self._last_call = time.time()


def use_cooldown(name, validator=None):
    global cooldown_table
    if name not in cooldown_table:
        cooldown_table[name] = CooldownPool()
    pool = cooldown_table[name]

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            with pool.lock:
                for attempt in range(1, pool.config.get_setting("RetryCount", 3) + 1):
                    try:
                        pool.wait()
                        result = func(*args, **kwargs)
                        success = True
                        if validator:
                            try:
                                success = validator(result)
                            except Exception as e:
                                print(f"{func.__name__}:{validator.__name__}: {e}")
                                success = False
                        pool.enforce(success=success)
                        if success:
                            return result
                        else:
                            print(f"{func.__name__}: Failed")
                    except Exception as e:
                        print(f"{func.__name__}: {e}")
                        pool.enforce(success=False)
            return None
        return wrapper
    return decorator


def truthy_validator(output: Any) -> bool:
    return True if output else False
