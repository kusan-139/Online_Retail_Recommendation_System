@echo off
title Online Retail Recommender - Menu

:: If not running in a new window, relaunch itself in cmd
if not defined IS_BATCH_RELAUNCH (
    set IS_BATCH_RELAUNCH=1
    start cmd /k "%~f0"
    exit /b
)

:: Activate venv (very important!)
call "%~dp0venv\Scripts\activate.bat"

:menu
cls
echo ============================================
echo   Online Retail Recommender - Choose Option
echo ============================================
echo.
echo   1. Run Auto-Analysis (Generate Reports)
echo   2. Run Streamlit Web App
echo   3. Exit
echo.
set /p choice=Enter choice (1-3): 

if "%choice%"=="1" goto analysis
if "%choice%"=="2" goto streamlit
if "%choice%"=="3" exit
goto menu

:analysis
cls
echo ===============================
echo Running Auto-Analysis...
echo ===============================
python retail_recommender.py
echo ===============================
echo Analysis Finished! Outputs saved in retail_outputs folder.
echo ===============================
pause
goto menu

:streamlit
cls
echo ===============================
echo Starting Streamlit App...
echo (Open http://localhost:8508 in your browser)
echo ===============================
streamlit run retail_recommender.py --server.port 8508
pause
goto menu
