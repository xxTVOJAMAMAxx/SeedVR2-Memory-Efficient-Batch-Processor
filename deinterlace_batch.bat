@echo off
setlocal enabledelayedexpansion

echo ========================================
echo Video Deinterlacer for SeedVR2
echo ========================================
echo.

REM Check if video file was dragged onto script
if "%~1"=="" (
    echo ERROR: No video file specified!
    echo.
    echo Usage: Drag and drop a video file onto this script
    echo OR run: deinterlace_video.bat "your_video.mp4"
    echo.
    pause
    exit /b 1
)

set "INPUT_VIDEO=%~1"
set "INPUT_DIR=%~dp1"
set "INPUT_NAME=%~n1"
set "OUTPUT_VIDEO=%INPUT_DIR%%INPUT_NAME%_deinterlaced_50fps.mp4"

echo Input video: %INPUT_NAME%%~x1
echo Output will be: %INPUT_NAME%_deinterlaced_50fps.mp4
echo.

REM Check if FFmpeg exists in current directory
if exist "ffmpeg.exe" (
    set "FFMPEG=ffmpeg.exe"
    goto :run_ffmpeg
)

REM Check if FFmpeg is in PATH
where ffmpeg >nul 2>&1
if %errorlevel%==0 (
    set "FFMPEG=ffmpeg"
    goto :run_ffmpeg
)

REM FFmpeg not found
echo FFmpeg not found!
echo.
echo Please download FFmpeg:
echo 1. Go to: https://www.gyan.dev/ffmpeg/builds/
echo 2. Download "ffmpeg-release-essentials.zip"
echo 3. Extract it
echo 4. Copy ffmpeg.exe to this folder: %~dp0
echo.
echo OR install via winget (if you have it):
echo    winget install Gyan.FFmpeg
echo.
pause
exit /b 1

:run_ffmpeg
echo Starting deinterlacing...
echo This may take several minutes depending on video length.
echo.

"%FFMPEG%" -i "%INPUT_VIDEO%" -vf yadif=mode=1:parity=auto:deint=all -r 50 -c:v libx264 -crf 18 -preset slow -c:a copy "%OUTPUT_VIDEO%"

if %errorlevel%==0 (
    echo.
    echo ========================================
    echo SUCCESS!
    echo ========================================
    echo.
    echo Deinterlaced video saved as:
    echo %OUTPUT_VIDEO%
    echo.
    echo You can now use this video with the batch processor!
    echo.
) else (
    echo.
    echo ========================================
    echo ERROR!
    echo ========================================
    echo.
    echo FFmpeg failed to process the video.
    echo Check the error messages above.
    echo.
)

pause
