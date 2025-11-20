import zstandard
import base64
import secrets
import subprocess
import json
import os


def deflate_to_base64(filename: str) -> str:
    with open(filename, "rb") as f:
        data: bytes = f.read()
    compressor = zstandard.ZstdCompressor(level=10)
    compressed = compressor.compress(data)
    return base64.b64encode(compressed).decode("ascii")


def inflate_to_file(data: str, filename: str) -> None:
    compressed: bytes = base64.b64decode(data)
    decompressor = zstandard.ZstdDecompressor()
    data: bytes = decompressor.decompress(compressed)
    with open(filename, "wb") as f:
        f.write(data)


def get_tmp_filename() -> str:
    return f"autosave{secrets.token_hex(4)}"


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
            "remglk": json.loads(stdout.strip()),
            "autosave.json": autosave_json,
            "autosave.glksave": autosave_glksave,
            "filerefs": filerefs,
        }

        self.delete_savefiles(fileref_list)

        return ret


class GlulxInterpreter(Interpreter):
    pass


class BocfelInterpreter(Interpreter):
    pass


if __name__ == "__main__":
    glulx_terp = GlulxInterpreter("C:\\Vhelas\\glulxe\\glulxe.exe", "..\\Alabaster.gblorb")
    output = glulx_terp.run()
    print("Turn 0: " + json.dumps(output, indent=4))

    glulx_terp_2 = GlulxInterpreter("C:\\Vhelas\\glulxe\\glulxe.exe", "..\\Alabaster.gblorb", {
        "autosave.json": output["autosave.json"],
        "autosave.glksave": output["autosave.glksave"],
    })
    output_2 = glulx_terp_2.run("no")
    print("Turn 1: " + json.dumps(output_2, indent=4))

    glulx_terp_3 = GlulxInterpreter("C:\\Vhelas\\glulxe\\glulxe.exe", "..\\Alabaster.gblorb", {
        "autosave.json": output_2["autosave.json"],
        "autosave.glksave": output_2["autosave.glksave"],
    })
    output_3 = glulx_terp_3.run("HELP")
    print("Turn 2: " + json.dumps(output_3, indent=4))
