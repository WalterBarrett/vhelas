import json
import os
import tempfile
import threading
from datetime import datetime, timedelta
from typing import Any


def encode_with_prefix(data: dict) -> dict:
    """Walk dict and prefix datetime keys before dumping."""
    out = {}
    for k, v in data.items():
        if isinstance(v, datetime):
            out[f"iso-time:{k}"] = v.isoformat()
        elif isinstance(v, dict):
            out[k] = encode_with_prefix(v)
        else:
            out[k] = v
    return out


def decode_with_prefix(data: dict) -> dict:
    """Walk dict and strip prefix, converting ISO strings back to datetime."""
    out = {}
    for k, v in data.items():
        if k.startswith("iso-time:") and isinstance(v, str):
            key = k[len("iso-time:"):]
            try:
                out[key] = datetime.fromisoformat(v)
            except ValueError:
                print(f"Error parsing \"{v}\" as a datetime!")
                out[key] = v
        elif isinstance(v, dict):
            out[k] = decode_with_prefix(v)
        else:
            out[k] = v
    return out


class PersistentDataStore:
    def __init__(self, name: str, compact: bool = False, timetolive: timedelta | None = None):
        self.path = f"stores/{name}.json"
        self._store = {}
        self.compact = compact
        self.timetolive = timetolive
        self._condition = threading.Condition()
        self._dirty = False
        self._closed = False
        try:
            if os.path.exists(self.path):
                with open(self.path, "r", encoding="utf-8") as f:
                    self._store = decode_with_prefix(json.load(f))
        except Exception:
            pass

    def save(self):
        if not self._dirty:
            return
        dir_name = os.path.dirname(self.path) or "."
        os.makedirs(dir_name, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(dir=dir_name)
        if self.timetolive:
            with self._condition:
                any_expired = False
                for key, value in list(self._store.items()):
                    if datetime.now() > value["updated"] + self.timetolive:
                        self._store.pop(key)
                        any_expired = True
                if any_expired:
                    self._condition.notify_all()
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                if self.compact:
                    json.dump(encode_with_prefix(self._store), f, separators=(",", ":"), ensure_ascii=False)
                else:
                    json.dump(encode_with_prefix(self._store), f, indent=4, sort_keys=True, ensure_ascii=False)
                f.write("\n")
            os.replace(tmp_path, self.path)
            self._dirty = False
        except Exception:
            try:
                os.remove(tmp_path)
            except OSError:
                pass
            raise

    def get(self, key: str, default: Any = None) -> Any | None:
        if self.timetolive:
            return self._store.get(key, {"value": default}).get("value", default)
        else:
            return self._store.get(key, default)

    def set(self, key: str, value: Any | None):
        with self._condition:
            if self.timetolive:
                self._store[key] = {"updated": datetime.now(), "value": value}
            else:
                self._store[key] = value
            self._dirty = True
            self._condition.notify_all()

    def delete(self, key: str):
        with self._condition:
            del self._store[key]
            self._dirty = True
            self._condition.notify_all()

    def pop(self, key: str, default: Any = None) -> Any | None:
        ret = self.get(key, default)
        if key in self._store:
            self.delete(key)
        return ret

    def pop_when_ready(self, key: str, default: Any = None, timeout: timedelta | int | None = None) -> Any | None:
        if not timeout:
            timeout = self.timetolive
        elif isinstance(timeout, int):
            timeout = timedelta(seconds=timeout)
        if not timeout:
            return self.pop(key, default)
        end_time = datetime.now() + timeout
        with self._condition:
            while key not in self._store and not self._closed:
                remaining = (end_time - datetime.now()).total_seconds()
                if remaining <= 0:  # timed out
                    return default
                self._condition.wait(timeout=min(1, remaining))
            if self._closed:
                return default
            return self.pop(key, default)


class PersistentDataStoreManager():
    def __init__(self, stores: dict, autosave_interval: int = 5):
        self._stores: dict[str, PersistentDataStore] = stores
        self._autosave_interval = autosave_interval
        self._stop_event = threading.Event()
        self._autosave_thread = threading.Thread(target=self._autosave_loop, daemon=True)
        self._autosave_thread.start()

    def get(self, key: str) -> PersistentDataStore:
        return self._stores.get(key)

    def _autosave_loop(self):
        while not self._stop_event.wait(self._autosave_interval):
            for store in self._stores.values():
                store.save()

    def shutdown(self):
        self._stop_event.set()
        for store in self._stores.values():
            store.save()
        self._autosave_thread.join()


_pds_lock = threading.Lock()
_pds = None


def get_stores() -> PersistentDataStoreManager:
    global _pds
    if _pds is None:
        with _pds_lock:
            if _pds is None:
                _pds = PersistentDataStoreManager({
                    "saves": PersistentDataStore("saves", timetolive=timedelta(minutes=10)),
                    "inputs": PersistentDataStore("inputs", timetolive=timedelta(minutes=10)),
                })
    return _pds


def close_stores() -> bool:
    global _pds
    if _pds:
        for store in _pds._stores.values():
            store._closed = True
            with store._condition:
                store._condition.notify_all()
        _pds.shutdown()
        return True
    return False
