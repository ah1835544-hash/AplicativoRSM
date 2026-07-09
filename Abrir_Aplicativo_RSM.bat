@echo off
REM ================================================================
REM  Abrir Aplicativo RSM - Optimizacion agroindustrial (UCE)
REM  Doble clic para iniciar. El navegador se abre solo.
REM  Para cerrar el aplicativo: cierre esta ventana negra.
REM ================================================================
title Aplicativo RSM (no cerrar mientras lo usa)

REM Ir a la carpeta donde esta este archivo .bat
cd /d "%~dp0"

REM Ruta del interprete de Python instalado
set "PYEXE=%LOCALAPPDATA%\Programs\Python\Python312\python.exe"

if not exist "%PYEXE%" (
    echo No se encontro Python en:
    echo   %PYEXE%
    echo Instale Python 3.12 o edite la ruta PYEXE en este archivo.
    echo.
    pause
    exit /b 1
)

echo Iniciando el Aplicativo RSM...
echo Se abrira su navegador en unos segundos.
echo Deje esta ventana ABIERTA mientras usa el aplicativo.
echo Para cerrar: cierre esta ventana.
echo.

"%PYEXE%" -m streamlit run app.py

REM Si streamlit termina con error, dejar ver el mensaje
pause
