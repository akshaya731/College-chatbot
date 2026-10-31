@echo off
title MIST Connect AI - Campus Chatbot
cd /d "%~dp0"
echo ========================================================
echo   Launching MIST Connect AI (Production VirtualEnv)
echo ========================================================
echo.
call .\venv\Scripts\activate.bat
streamlit run app.py --server.port 8501
pause

