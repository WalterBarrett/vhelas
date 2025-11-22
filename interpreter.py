import subprocess
import json
import os
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


def get_output_from_remglk_entries(remglk: list[dict], input: str | None, fallback_windows: list[dict] = None):
    output_buffer = []
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
                                    text = inner.get("text", "")
                                    match style:
                                        case "emphasized":
                                            output_buffer.append(f"*{text}*")
                                        case "preformatted":
                                            output_buffer.append(f"`{text}`")
                                        case "header":
                                            output_buffer.append(f"# {text}")
                                        case "subheader":
                                            output_buffer.append(f"## {text}")
                                        case "alert":
                                            output_buffer.append(f"**{text}**")
                                        case "note":
                                            output_buffer.append(f"*{text}*")
                                        case "blockquote":
                                            output_buffer.append(f"> {text}")
                                        case "input":
                                            if text.strip() != input.strip():
                                                output_buffer.append(f"**`{text}`**")
                                        case "user1":
                                            output_buffer.append(f"*{text}*")
                                        case "user2":
                                            output_buffer.append(f"*{text}*")
                                        case "normal" | "unknown" | _:
                                            output_buffer.append(text)
                            output_buffer.append("\n")
        elif type == "error":
            output_buffer.append(f"[Error: {rgo.get('message', 'Unspecified error.')}]\n")

    # print(json.dumps(remglk))
    return "".join(output_buffer).rstrip("> \t\n\r")


class Interpreter(ABC):
    """Interactive Fiction interpreter."""
    def __init__(self, path: str, gamename: str, messages: list[dict] = None, savedata: dict = None):
        self.path = path
        self.gamename = gamename
        self.messages = messages if messages is not None else []
        self.savedata = savedata if savedata is not None else {}

    @abstractmethod
    def __call__(self, input: str = None, previousInputs: list[str] = None) -> tuple[dict, str, str]:
        pass


class RemGlkGlulxeInterpreter(Interpreter):
    """RemGlk-Glulxe interpreter. Uses autosaves to maintain VM state across messages."""
    def __init__(self, path: str, gamename: str, messages: list[dict] = None, savedata: dict = None):
        super().__init__(path, gamename, messages, savedata)
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
        args.append(self.gamename)
        return args

    def _write_savefiles(self) -> None:
        if "autosave.json" in self.savedata:
            with open(f"{self.savename}.json", "w") as f:
                f.write(json.dumps(self.savedata["autosave.json"]))
        if "autosave.glksave" in self.savedata:
            inflate_to_file(self.savedata["autosave.glksave"], f"{self.savename}.glksave")

        for fileref, filecontents in self.savedata.get("filerefs", {}).items():
            if filecontents:
                inflate_to_file(filecontents, fileref)

    def _get_input_parameters(self) -> tuple[int, str]:
        if "autosave.json" in self.savedata:
            if "windows" in self.savedata["autosave.json"]:
                windows = self.savedata["autosave.json"]["windows"]
                for window in windows:
                    if window.get("line_request", 0) or window.get("line_request_uni", 0):
                        return window["tag"], "line"
                for window in windows:
                    if window.get("char_request", 0) or window.get("char_request_uni", 0):
                        return window["tag"], "char"
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

    def __call__(self, input: str = None, previousInputs: list[str] = None) -> tuple[dict, str, str]:
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
            if not input.strip():
                match input_type:
                    case "line":
                        input = "wait"
                    case "char":
                        input = " "
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

        with open(f"{self.savename}.json", "r") as f:
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

        return ret, input, get_output_from_remglk_entries(remglk, input, autosave_json.get("windows", {}))


if __name__ == "__main__":
    def test_glulxe():
        print("Turn 0: ")
        glulx_terp = RemGlkGlulxeInterpreter("C:\\Vhelas\\remglk-terps\\terps\\glulxe\\glulxe.exe", "..\\Alabaster.gblorb")
        output, _, _ = glulx_terp()
        print("Contents: " + json.dumps(output, indent=4))

        print("Turn 1: ")
        glulx_terp_2 = RemGlkGlulxeInterpreter("C:\\Vhelas\\remglk-terps\\terps\\glulxe\\glulxe.exe", "..\\Alabaster.gblorb", {
            "autosave.json": output["autosave.json"],
            "autosave.glksave": output["autosave.glksave"],
            "filerefs": output["filerefs"],
        })
        output_2, _, _ = glulx_terp_2("no")
        print("Contents: " + json.dumps(output_2, indent=4))

        print("Turn 2: ")
        glulx_terp_3 = RemGlkGlulxeInterpreter("C:\\Vhelas\\remglk-terps\\terps\\glulxe\\glulxe.exe", "..\\Alabaster.gblorb", {
            "autosave.json": output_2["autosave.json"],
            "autosave.glksave": output_2["autosave.glksave"],
            "filerefs": output_2["filerefs"],
        })
        output_3, _, _ = glulx_terp_3("HELP")
        print("Contents: " + json.dumps(output_3, indent=4))

        print("Turn 3: ")
        glulx_terp_4 = RemGlkGlulxeInterpreter("C:\\Vhelas\\remglk-terps\\terps\\glulxe\\glulxe.exe", "..\\Alabaster.gblorb", {
            "autosave.json": output_3["autosave.json"],
            "autosave.glksave": output_3["autosave.glksave"],
            "filerefs": output_3["filerefs"],
        })
        output_4, _, _ = glulx_terp_4(" ")
        print("Contents: " + json.dumps(output_4, indent=4))

    test_glulxe()
