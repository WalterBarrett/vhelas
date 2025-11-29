import signal
import subprocess
import json
import os
import re
from abc import ABC, abstractmethod
from utils import deflate_to_base64, get_tmp_filename, inflate_to_file


def json_array_from_stdout(data: str) -> list[dict]:
    decoder = json.JSONDecoder()
    idx = 0
    results = []
    length = len(data)

    while idx < length:
        while idx < length and data[idx].isspace():
            idx += 1
        if idx >= length:
            break
        obj, end = decoder.raw_decode(data, idx)
        results.append(obj)
        idx = end

    return results


def get_status_window_from_glk(windows) -> int | None:
    if windows:
        for window in windows:
            win_type = window.get("type", None)
            if (win_type == "grid" or win_type == 4) and window.get("gridwidth", None) == 240 and window.get("gridheight", 0):
                return window.get("id", window.get("tag", None))
    return None


def get_status_from_remglk_entries(remglk: list[dict], fallback_windows: list[dict] | None = None) -> list[str] | None:
    # TODO: This doesn't properly handle status lines that mix styles (not that I have any examples at hand)
    status_window = get_status_window_from_glk(fallback_windows)
    status = None
    last_status_rgo = None
    for rgo in remglk:
        type = rgo.get("type", None)
        if type == "error":
            continue
        elif type == "update":
            windows = rgo.get("windows", fallback_windows)
            if windows:
                new_status_window = get_status_window_from_glk(windows)
                if new_status_window is not None:
                    status_window = new_status_window
            content = rgo.get("content", [])
            if content:
                for window in content:
                    window_id = window.get("id", None)
                    if window_id == status_window:
                        for line in window.get("lines", None):
                            inner_content = line.get("content", {})
                            if inner_content:
                                for inner in inner_content:
                                    # style = inner.get("style", None)
                                    if last_status_rgo != rgo:
                                        status = None
                                        last_status_rgo = rgo
                                    text = inner.get("text", "")
                                    new_status = [text[0:80].strip(), text[81:160].strip(), text[161:240].strip()]
                                    if status:
                                        status = [f"{status[0]}\n{new_status[0]}", f"{status[1]}\n{new_status[1]}", f"{status[2]}\n{new_status[2]}"]
                                    else:
                                        status = new_status
    if status:
        status = [status[0].strip(), status[1].strip(), status[2].strip()]
    return status


def get_output_from_remglk_entries(remglk: list[dict], input: str | None, fallback_windows: list[dict] | None = None):
    output_buffer = []
    status = get_status_from_remglk_entries(remglk, fallback_windows)
    for rgo in remglk:
        type = rgo.get("type", None)
        if type == "update":
            windows = rgo.get("windows", fallback_windows)
            window_ids = set()
            for window in windows:
                window_type = window.get("type", None)
                if window_type == "buffer" or window_type == 3:
                    window_ids.add(window.get("id", window.get("tag", None)))
            content = rgo.get("content", [])
            if content:
                for window in content:
                    window_id = window.get("id", None)
                    if window_id in window_ids:
                        for text in window.get("text", None):
                            inner_content = text.get("content", {})
                            if inner_content:
                                for inner in inner_content:
                                    style = inner.get("style", None)
                                    innertext = inner.get("text", "")
                                    match style:
                                        case "emphasized":
                                            output_buffer.append(f"*{innertext}*")
                                        case "preformatted":
                                            output_buffer.append(f"`{innertext}`")
                                        case "header":
                                            output_buffer.append(f"# {innertext}")
                                        case "subheader":
                                            output_buffer.append(f"## {innertext}")
                                        case "alert":
                                            output_buffer.append(f"**{innertext}**")
                                        case "note":
                                            output_buffer.append(f"*{innertext}*")
                                        case "blockquote":
                                            output_buffer.append(f"> {innertext}")
                                        case "input":
                                            if input and innertext.strip() != input.strip():
                                                output_buffer.append(f"**`{innertext}`**")
                                        case "user1":
                                            output_buffer.append(f"*{innertext}*")
                                        case "user2":
                                            output_buffer.append(f"*{innertext}*")
                                        case "normal" | "unknown" | _:
                                            output_buffer.append(innertext)
                            output_buffer.append("\n")
        elif type == "error":
            output_buffer.append(f"[Error: {rgo.get('message', 'Unspecified error.')}]\n")

    if status is not None:
        output_buffer.insert(0, f"<!--STATUS:{json.dumps(status)}-->")

    return "".join(output_buffer).rstrip("> \t\n\r")


def get_input_parameters_from_glk(windows) -> tuple[int, str, int | None]:
    """Returns input window, input method, and status window (if detected)."""
    if windows:
        for window in windows:
            if window.get("line_request", 0) or window.get("line_request_uni", 0):
                return window.get("id", window.get("tag", 0)), "line"
        for window in windows:
            if window.get("char_request", 0) or window.get("char_request_uni", 0):
                return window.get("id", window.get("tag", 0)), "char"
        for window in windows:
            win_type = window.get("type", None)
            if win_type == "buffer" or win_type == 3:
                return window.get("id", window.get("tag", 0)), "line"
    return 0, "line"


def normalize_remglk_input(input: str, input_type: str) -> str:
    def remove_comment(match):
        return ""
    input = re.sub(r"<!--(.*?)-->", remove_comment, input, flags=re.DOTALL).strip()
    if not input.strip():
        match input_type:
            case "line":
                input = "wait"
            case "char":
                input = " "
    return input


class Interpreter(ABC):
    """Interactive Fiction interpreter."""
    def __init__(self, path: str, gamename: str, messages: list[dict] = None, savedata: dict = None, extraargs: list[str] = None):
        self.path = path
        self.gamename = gamename
        self.messages = messages if messages is not None else []
        self.savedata = savedata if savedata is not None else {}
        self.extraargs = extraargs if extraargs is not None else []

    @abstractmethod
    def __call__(self, input: str = None, previousInputs: list[str] | None = None) -> tuple[dict, str, str]:
        pass


class RemGlkGlulxeInterpreter(Interpreter):
    """RemGlk-Glulxe interpreter. Uses autosaves to maintain VM state across messages."""
    def __init__(self, path: str, gamename: str, messages: list[dict] = None, savedata: dict = None, extraargs: list[str] = None):
        super().__init__(path, gamename, messages, savedata, extraargs)
        self.savename = get_tmp_filename()
        self.autorestore = "autosave.json" in self.savedata

    def _get_generation(self) -> int:
        if "autosave.json" in self.savedata:
            if "generation" in self.savedata["autosave.json"]:
                return self.savedata["autosave.json"]["generation"]
        return 1

    def _get_parameters(self) -> list[str]:
        args = [self.path, "-fm", "-width", "240", "-height", "240", "--autosave", "-singleturn", "--autoname", self.savename]
        # For some reason, -stderr doesn't seem to work correctly.
        if self.autorestore:
            args.append("--autorestore")
        if self.extraargs:
            args.extend(self.extraargs)
        args.append(self.gamename)
        return args

    def _write_savefiles(self) -> None:
        if "autosave.json" in self.savedata:
            with open(f"{self.savename}.json", "w", encoding="utf-8") as f:
                f.write(json.dumps(self.savedata["autosave.json"]))
        if "autosave.glksave" in self.savedata:
            inflate_to_file(self.savedata["autosave.glksave"], f"{self.savename}.glksave")

        for fileref, filecontents in self.savedata.get("filerefs", {}).items():
            if filecontents:
                inflate_to_file(filecontents, fileref)

    def _get_input_parameters(self) -> tuple[int, str]:
        if "autosave.json" in self.savedata:
            return get_input_parameters_from_glk(self.savedata["autosave.json"]["windows"])
        return 0, "line"

    def _get_fileref_list(self, autosavejson: list[dict]) -> list[str]:
        filerefs = autosavejson.get("filerefs", [])
        if not filerefs:
            return []
        return [fileref.get("filename") for fileref in filerefs if "filename" in fileref]

    def _delete_savefiles(self, new_filerefs) -> None:
        if os.path.exists(f"{self.savename}.json"):
            os.remove(f"{self.savename}.json")
        if os.path.exists(f"{self.savename}.glksave"):
            os.remove(f"{self.savename}.glksave")
        if os.path.exists(f"{self.savename}.undos"):
            os.remove(f"{self.savename}.undos")

        old_filerefs = self._get_fileref_list(self.savedata.get("autosave.json", {}))

        filerefs = list(set(old_filerefs) | set(new_filerefs))
        for fileref in filerefs:
            if os.path.exists(fileref):
                os.remove(fileref)

    def __call__(self, input: str = None, previousInputs: list[str] | None = None) -> tuple[dict, str, str]:
        if self.autorestore:
            self._write_savefiles()

        process = subprocess.Popen(
            self._get_parameters(),
            stdin=subprocess.PIPE if self.autorestore else subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        if self.autorestore:
            window_number, input_type = self._get_input_parameters()
            input = normalize_remglk_input(input, input_type)
            stdout, _ = process.communicate(input=json.dumps({
                "type": input_type,
                "gen": self._get_generation(),
                "window": window_number,
                "value": input,
            }))
        else:
            stdout, _ = process.communicate()

        if process.returncode != 0:
            raise RuntimeError(f"Process \"{' '.join(self._get_parameters())}\" failed: {stdout.strip()}")

        with open(f"{self.savename}.json", "r", encoding="utf-8") as f:
            autosave_json = json.load(f)
        autosave_glksave = deflate_to_base64(f"{self.savename}.glksave")

        filerefs = {}
        fileref_list = self._get_fileref_list(autosave_json)
        for fileref in fileref_list:
            if os.path.exists(fileref):
                filerefs[fileref] = deflate_to_base64(fileref)
            else:
                filerefs[fileref] = None

        remglk = json_array_from_stdout(stdout.strip())
        ret = {
            "autosave.json": autosave_json,
            "autosave.glksave": autosave_glksave,
            "filerefs": filerefs,
        }

        self._delete_savefiles(fileref_list)

        final_output = get_output_from_remglk_entries(remglk, input, autosave_json.get("windows", {}))

        if previousInputs is None:
            final_output = f"<!--GAMESTART:true-->{final_output}"

        return ret, input, final_output


class RemGlkInterpreter(Interpreter):
    """Generic RemGlk interpreter. Must re-run all provided commands to return to current the current state."""
    def __init__(self, path: str, gamename: str, messages: list[dict] = None, savedata: dict = None, extraargs: list[str] = None):
        super().__init__(path, gamename, messages, savedata, extraargs)

    def _get_parameters(self) -> list[str]:
        args = [self.path, "-fm", "-width", "240", "-height", "240"]
        # For some reason, -stderr doesn't seem to work correctly.
        if self.extraargs:
            args.extend(self.extraargs)
        args.append(self.gamename)
        return args

    def _wait_for_update(self, responses, current_generation: int, current_windows: dict = None) -> tuple[list[dict], int, dict]:
        messages = []
        windows = current_windows
        generation = current_generation
        for obj in responses:
            messages.append(obj)
            if isinstance(obj, dict) and obj.get("type") == "update":
                for msg in messages:
                    if "windows" in msg:
                        windows = msg["windows"]
                    if "gen" in msg:
                        generation = msg["gen"]
                return messages, generation, windows
        raise Exception("Subprocess closed unexpectedly.")

    def __call__(self, input: str = None, previousInputs: list[str] | None = None) -> tuple[dict, str, str]:
        def iter_json_stream(stream):
            decoder = json.JSONDecoder()
            buffer = ""
            while True:
                chunk = stream.read(1)  # TODO: Make this streaming work better. This is absolutely churning through string allocations and JSON decodes and such.
                if not chunk:
                    break
                buffer += chunk
                while buffer:
                    buffer = buffer.lstrip()
                    try:
                        obj, idx = decoder.raw_decode(buffer)
                        yield obj
                        buffer = buffer[idx:].lstrip()
                    except json.JSONDecodeError:
                        break  # Not enough data yet

        def send_json(proc, obj):
            msg = json.dumps(obj)
            proc.stdin.write(msg + "\n")
            proc.stdin.flush()

        process = subprocess.Popen(
            self._get_parameters(),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        responses = iter_json_stream(process.stdout)

        messages, generation, windows = self._wait_for_update(responses, 0)
        if previousInputs is None:
            return {}, None, f"<!--GAMESTART:true-->{get_output_from_remglk_entries(messages, None, windows)}"

        for cmd in previousInputs + [input]:
            window_number, input_type = get_input_parameters_from_glk(windows)
            input = normalize_remglk_input(cmd, input_type)
            json_thing = {
                "type": input_type,
                "gen": generation,
                "window": window_number,
                "value": input,
            }
            send_json(process, json_thing)
            messages, generation, windows = self._wait_for_update(responses, generation, windows)

        process.send_signal(signal.SIGTERM)
        process.wait()

        return {}, input, get_output_from_remglk_entries(messages, None, windows)
