@echo off

REM -----------------------------------------------------------
REM BAT for UNC path with auto-drive letter from pushd
REM -----------------------------------------------------------

REM pushd maps UNC path to a temp drive letter
pushd "\\DC1\data\adeyemi.aderoju\Desktop\Ade\Python Project\Transaction Monitoring"

REM Activate virtual env using this drive letter context
call .venv\Scripts\activate.bat

REM Generate timestamp
for /f %%i in ('powershell -Command "Get-Date -Format yyyy-MM-dd_HH-mm-ss"') do set timestamp=%%i

REM Make sure logs folder exists
if not exist logs mkdir logs

REM Run Python and log output
python -m utility.run_checks > logs\run_log_%timestamp%.txt 2>&1

echo Log saved: logs\run_log_%timestamp%.txt

REM Clean up the mapped drive
popd

pause