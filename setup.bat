@echo off
rem Richtet die Python-Umgebung (.venv) fuer Schichtdicke.py ein.
rem Einmalig per Doppelklick ausfuehren; danach mit start.bat starten.
setlocal
cd /d "%~dp0"

rem --- Passendes Python suchen (3.11 bis 3.13) ---
set "PY="
for %%V in (3.12 3.13 3.11) do (
    if not defined PY (
        py -%%V -c "pass" >nul 2>&1 && set "PY=py -%%V"
    )
)
if not defined PY (
    python -c "import sys; sys.exit(0 if (3,11) <= sys.version_info[:2] <= (3,13) else 1)" >nul 2>&1 && set "PY=python"
)
if not defined PY (
    echo Kein passendes Python gefunden. Benoetigt wird Python 3.11, 3.12 oder 3.13.
    echo Download: https://www.python.org/downloads/
    echo Alternativ in einer Konsole:  py install 3.12
    pause
    exit /b 1
)
echo Verwende: %PY%

rem --- Alte .venv entfernen (kann von einem anderen PC stammen) ---
if exist .venv (
    echo Entferne vorhandene .venv ...
    rmdir /s /q .venv
)

echo Erstelle .venv ...
%PY% -m venv .venv || goto :fehler

set "VPY=.venv\Scripts\python.exe"
"%VPY%" -m pip install --upgrade pip setuptools wheel || goto :fehler
"%VPY%" -m pip install -r requirements.txt || goto :fehler
"%VPY%" install_pycorrelate.py || goto :fehler

echo.
echo Fertig. Das Programm startet jetzt mit start.bat.
pause
exit /b 0

:fehler
echo.
echo Einrichtung fehlgeschlagen, siehe Meldungen oben.
pause
exit /b 1
