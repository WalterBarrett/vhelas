import os
import threading
import time
from typing import Any

import yaml
from watchdog.events import FileSystemEventHandler


class WrapperStore():
    def __init__(self, filename, dependents=[], validator=None):
        self.filename = filename
        self.dependents = dependents
        self.validator = validator
        self.store = {}

    def load(self):
        with open(self.filename, "r") as f:
            tmp = yaml.safe_load(f)
            if tmp and self.validator:
                tmp = self.validator(tmp)
            if tmp:
                self.store = tmp
                return True
        return False


class WrapperConfiguration:
    def __init__(self):
        def model_validator(model_data):
            # Remove models that require keys we don't have in the keyring
            for key, value in list(model_data.items()):
                if not value["keychain"] in self.stores["keys.yaml"].store.keys():
                    model_data.pop(key)
            return model_data

        self.stores = {
            "settings.yaml": WrapperStore("settings.yaml"),
            "keys.yaml": WrapperStore("keys.yaml", dependents=["models.yaml"]),
            "models.yaml": WrapperStore("models.yaml", validator=model_validator),
        }

        for store in self.stores.values():
            store.load()

    def has_store(self, filename):
        return filename in self.stores

    def load_by_filename(self, filename):
        return self.stores[filename].load()

    def get_dependents(self, filename):
        return self.stores[filename].dependents

    def get_key(self, key: str) -> str | None:
        return self.stores["keys.yaml"].store.get(key, None)

    def get_model(self, key: str) -> dict[str, Any] | None:
        return self.stores["models.yaml"].store.get(key, None)

    def get_models(self):
        return self.stores["models.yaml"].store.items()

    def get_setting(self, key: str, default: Any = None):
        return self.stores["settings.yaml"].store.get(key, default)


class ReloadHandler(FileSystemEventHandler):
    def __init__(self):
        self.config = get_config()
        self.last_reload = {}

    def on_modified(self, event):
        filename = os.path.basename(event.src_path)
        now = time.time()
        if now - self.last_reload.get(filename, 0) < 0.2:
            return
        self.last_reload[filename] = now
        try:
            if self.config.has_store(filename):
                if self.config.load_by_filename(filename):
                    print(f"Successfully reloaded {filename}.")
                else:
                    print(f"Error reloading {filename}.")
                for dependent in self.config.get_dependents(filename):
                    if self.config.load_by_filename(dependent):
                        print(f"Successfully reloaded {dependent}.")
                    else:
                        print(f"Error reloading {dependent}.")
        except Exception as e:
            print(f"Error loading {filename}: {e}")


_config_lock = threading.Lock()
_config = None


def get_config() -> WrapperConfiguration:
    global _config
    if _config is None:
        with _config_lock:
            if _config is None:
                _config = WrapperConfiguration()
    return _config
