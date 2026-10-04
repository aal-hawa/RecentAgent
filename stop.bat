@echo off
title RA-Stopper
echo Stopping RecentAgent (Streamlit on port 8501)...

set FOUND=0

rem 1) Close the terminal window that runs this app's start.bat.
rem    Matched by COMMAND LINE, never by window title: a title match like
rem    "RecentAgent*" could also kill Explorer / VS Code / other apps whose
rem    window merely shows the folder name.
rem    NOTE: no WMI -Filter (its quotes get mangled through cmd layers) -
rem    filter in Where-Object instead, which is quoting-safe.
powershell -NoProfile -Command "$q='Recent'+'Agent'; $p='start'+'.bat'; Get-CimInstance Win32_Process | Where-Object { $_.ProcessId -ne $PID -and $_.Name -eq 'cmd.exe' -and $_.CommandLine -like ('*' + $q + '*' + $p + '*') } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }" >nul 2>&1

rem 2) Kill anything still listening on the app port, with its process tree
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :8501 ^| findstr LISTENING') do (
    taskkill /F /T /PID %%a >nul 2>&1
    set FOUND=1
)

rem 3) Sweep: leftover python/streamlit belonging to THIS folder (match by
rem    executable path - cmdlines are relative like ".venv\Scripts\python.exe")
powershell -NoProfile -Command "$q='Recent'+'Agent'; Get-CimInstance Win32_Process | Where-Object { $_.ProcessId -ne $PID -and ($_.Name -eq 'python.exe' -or $_.Name -eq 'streamlit.exe') -and (($_.ExecutablePath -like ('*'+$q+'*')) -or ($_.CommandLine -like ('*'+$q+'*'))) -and $_.CommandLine -like '*streamlit*' } | ForEach-Object { taskkill /F /T /PID $($_.ProcessId) 2>&1 | Out-Null }" >nul 2>&1

if %FOUND%==1 (
    ping -n 2 127.0.0.1 >nul
    netstat -ano | findstr :8501 | findstr LISTENING >nul
    if errorlevel 1 (
        echo RecentAgent stopped - server and terminal window are closed.
    ) else (
        echo RecentAgent process was killed but port 8501 is still busy - check other apps.
    )
) else (
    echo RecentAgent is not running.
)
ping -n 3 127.0.0.1 >nul
