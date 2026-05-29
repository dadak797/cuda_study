@echo off
setlocal

set "ROOT=%~dp0"
set "ROOT=%ROOT:~0,-1%"
set "TARGET=hello_cuda"
set "BUILD_DIR=%ROOT%\build"

set "CONFIG=%~1"
if "%CONFIG%"=="" set "CONFIG=Debug"

if /I "%CONFIG%"=="debug" (
  set "CONFIG=Debug"
) else if /I "%CONFIG%"=="release" (
  set "CONFIG=Release"
) else (
  echo Usage: %~nx0 [debug^|release]
  echo.
  echo Examples:
  echo   %~nx0 debug
  echo   %~nx0 release
  exit /b 1
)

echo [configure] %CONFIG%
cmake -S "%ROOT%" -B "%BUILD_DIR%" -DCMAKE_SUPPRESS_REGENERATION=ON
if not "%ERRORLEVEL%"=="0" exit /b %ERRORLEVEL%

echo [build] %CONFIG%
cmake --build "%BUILD_DIR%" --config "%CONFIG%"
if not "%ERRORLEVEL%"=="0" exit /b %ERRORLEVEL%

set "EXE=%BUILD_DIR%\%CONFIG%\%TARGET%.exe"
if not exist "%EXE%" (
  set "EXE=%BUILD_DIR%\%TARGET%.exe"
)

if not exist "%EXE%" (
  echo Executable not found for %CONFIG%.
  echo Expected:
  echo   %BUILD_DIR%\%CONFIG%\%TARGET%.exe
  echo   %BUILD_DIR%\%TARGET%.exe
  exit /b 1
)

echo [run] "%EXE%"
"%EXE%"
exit /b %errorlevel%
