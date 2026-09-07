@echo off

cd /d "%~dp0"
cd ..

for %%f in ("notebooks\*.ipynb") do (
    echo Konwertowanie %%~nxf...

    ".venv\Scripts\python.exe" -m jupyter nbconvert --to script "%%f" --output-dir "notebooks"

    powershell -NoProfile -Command "(Get-Content 'notebooks\%%~nf.py') | Where-Object { $_ -notmatch 'get_ipython\(\)' } | Set-Content 'notebooks\%%~nf.py'"
)

echo.
echo Konwersja zakonczona.
pause
