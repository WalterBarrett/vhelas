import subprocess
import json
import os
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


class Interpreter:
    """RemGlk-based interpreter."""
    def __init__(self, path: str, gamename: str, savedata: dict = {}):
        self.path = path
        self.gamename = gamename
        self.savedata = savedata
        self.savename = get_tmp_filename()
        self.autorestore = "autosave.json" in self.savedata
        pass

    def get_generation(self) -> int:
        if "autosave.json" in self.savedata:
            if "generation" in self.savedata["autosave.json"]:
                return self.savedata["autosave.json"]["generation"]
        return 1

    def get_parameters(self) -> list[str]:
        args = [self.path, "-fm", "-width", "240", "-height", "240", "--autosave", "-singleturn", "--autoname", self.savename]
        # For some reason, -stderr doesn't seem to work correctly.
        if self.autorestore:
            args.append("--autorestore")
        args.append(self.gamename)
        return args

    def write_savefiles(self) -> None:
        if "autosave.json" in self.savedata:
            with open(f"{self.savename}.json", "w") as f:
                f.write(json.dumps(self.savedata["autosave.json"]))
        if "autosave.glksave" in self.savedata:
            inflate_to_file(self.savedata["autosave.glksave"], f"{self.savename}.glksave")

        for fileref, filecontents in self.savedata.get("filerefs", {}).items():
            if filecontents:
                inflate_to_file(filecontents, fileref)

    def get_input_parameters(self) -> tuple[int, str]:
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

    def get_fileref_list(self, autosavejson: list[dict]) -> list[str]:
        filerefs = autosavejson.get("filerefs", [])
        if not filerefs:
            return []
        return [fileref.get("filename") for fileref in filerefs if "filename" in fileref]

    def delete_savefiles(self, new_filerefs) -> None:
        if os.path.exists(f"{self.savename}.json"):
            os.remove(f"{self.savename}.json")
        if os.path.exists(f"{self.savename}.glksave"):
            os.remove(f"{self.savename}.glksave")
        if os.path.exists(f"{self.savename}.undos"):
            os.remove(f"{self.savename}.undos")

        old_filerefs = self.get_fileref_list(self.savedata.get("autosave.json", {}))

        filerefs = list(set(old_filerefs) | set(new_filerefs))
        for fileref in filerefs:
            if os.path.exists(fileref):
                os.remove(fileref)

    def run(self, input: str = None) -> dict:
        if self.autorestore:
            self.write_savefiles()

        process = subprocess.Popen(
            self.get_parameters(),
            stdin=subprocess.PIPE if self.autorestore else subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        if self.autorestore:
            window_number, input_type = self.get_input_parameters()
            if not input.strip():
                match input_type:
                    case "line":
                        input = "wait"
                    case "char":
                        input = " "
            stdout, _ = process.communicate(input=json.dumps({
                "type": input_type,
                "gen": self.get_generation(),
                "window": window_number,
                "value": input,
            }))
        else:
            stdout, _ = process.communicate()

        if process.returncode != 0:
            raise RuntimeError(f"Process \"{' '.join(self.get_parameters())}\" failed: {stdout.strip()}")

        with open(f"{self.savename}.json", "r") as f:
            autosave_json = json.load(f)
        autosave_glksave = deflate_to_base64(f"{self.savename}.glksave")

        filerefs = {}
        fileref_list = self.get_fileref_list(autosave_json)
        for fileref in fileref_list:
            if os.path.exists(fileref):
                filerefs[fileref] = deflate_to_base64(fileref)
            else:
                filerefs[fileref] = None

        ret = {
            "remglk": json_array_from_stdout(stdout.strip()),
            "autosave.json": autosave_json,
            "autosave.glksave": autosave_glksave,
            "filerefs": filerefs,
        }

        self.delete_savefiles(fileref_list)

        return ret


class GlulxeInterpreter(Interpreter):
    pass


if __name__ == "__main__":
    def test_glulxe():
        print("Turn 0: ")
        glulx_terp = GlulxeInterpreter("C:\\Vhelas\\remglk-terps\\terps\\glulxe\\glulxe.exe", "..\\Alabaster.gblorb")
        output = glulx_terp.run()
        print("Contents: " + json.dumps(output, indent=4))

        print("Turn 1: ")
        glulx_terp_2 = GlulxeInterpreter("C:\\Vhelas\\remglk-terps\\terps\\glulxe\\glulxe.exe", "..\\Alabaster.gblorb", {
            "autosave.json": output["autosave.json"],
            "autosave.glksave": output["autosave.glksave"],
            "filerefs": output["filerefs"],
        })
        output_2 = glulx_terp_2.run("no")
        print("Contents: " + json.dumps(output_2, indent=4))

        print("Turn 2: ")
        glulx_terp_3 = GlulxeInterpreter("C:\\Vhelas\\remglk-terps\\terps\\glulxe\\glulxe.exe", "..\\Alabaster.gblorb", {
            "autosave.json": output_2["autosave.json"],
            "autosave.glksave": output_2["autosave.glksave"],
            "filerefs": output_2["filerefs"],
        })
        output_3 = glulx_terp_3.run("HELP")
        print("Contents: " + json.dumps(output_3, indent=4))

        print("Turn 3: ")
        glulx_terp_4 = GlulxeInterpreter("C:\\Vhelas\\remglk-terps\\terps\\glulxe\\glulxe.exe", "..\\Alabaster.gblorb", {
            "autosave.json": output_3["autosave.json"],
            "autosave.glksave": output_3["autosave.glksave"],
            "filerefs": output_3["filerefs"],
        })
        output_4 = glulx_terp_4.run(" ")
        print("Contents: " + json.dumps(output_4, indent=4))

    test_glulxe()
