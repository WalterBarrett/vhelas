# Vhelas

A connection wrapper for playing Interactive Fiction games via SillyTavern (with use of [the appropriate SillyTavern extension](https://github.com/WalterBarrett/vhelas-status-line)). This is currently an early work-in-progress, not intended for use by anyone other than myself at the moment, thus the incomplete instructions.

## Setup

These instructions are a work-in-progress.

### Create the venv

```bash
python -m venv venv
```

### Activate the venv

- **Linux:** `source venv/bin/activate`
- **Powershell:** `.\venv\Scripts\Activate.ps1`
- **Windows Command Prompt:** `venv\Scripts\activate`

### Install the prerequisites

```bash
pip install llama-index-core llama-index-llms-openai-like fastapi uvicorn watchdog zstandard mistune pillow
```

### Setting up remglk-terps

```bash
git clone --recursive -b remglk-terps https://github.com/WalterBarrett/garglk.git remglk-terps
cd remglk-terps
make
```

## Licenses

This repository is under [the 3-Clause BSD License](License.md). We utilize libraries with [licenses](LibraryLicenses.md) under a mix of the MIT License (FastAPI, LlamaIndex), the MIT-CMU License (Pillow), and the 3-Clause BSD License (Uvicorn, Mistune, Zstandard).
