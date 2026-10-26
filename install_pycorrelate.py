"""
Installiert pycorrelate 0.3 in die aktuelle Python-Umgebung.

Das Paket auf PyPI nutzt ein veraltetes versioneer.py, das ab Python 3.12
abbricht (configparser.SafeConfigParser / readfp wurden entfernt). Dieses
Skript lädt den Quellcode von PyPI, ersetzt die beiden Aufrufe und
installiert das Paket anschließend mit pip. Nur Standardbibliothek.

Aufruf: python install_pycorrelate.py   (wird von setup.bat ausgeführt)
"""

import json
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

VERSION = "0.3"


def main():
    with urllib.request.urlopen(f"https://pypi.org/pypi/pycorrelate/{VERSION}/json") as resp:
        info = json.load(resp)
    url = next(u["url"] for u in info["urls"] if u["packagetype"] == "sdist")

    with tempfile.TemporaryDirectory() as tmp:
        archive = Path(tmp) / "pycorrelate.tar.gz"
        urllib.request.urlretrieve(url, archive)
        with tarfile.open(archive) as tar:
            tar.extractall(tmp, filter="data")

        src = Path(tmp) / f"pycorrelate-{VERSION}"
        versioneer = src / "versioneer.py"
        text = versioneer.read_text(encoding="utf-8")
        text = text.replace("configparser.SafeConfigParser()", "configparser.ConfigParser()")
        text = text.replace("parser.readfp(f)", "parser.read_file(f)")
        versioneer.write_text(text, encoding="utf-8")

        subprocess.check_call([sys.executable, "-m", "pip", "install", str(src)])


if __name__ == "__main__":
    main()
