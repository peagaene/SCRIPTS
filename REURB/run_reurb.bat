@echo off
setlocal
cd /d %~dp0
call "C:\Users\compartilhar\anaconda3\Scripts\activate.bat" geo_env
python -m reurb.main
endlocal
