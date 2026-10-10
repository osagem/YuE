@echo off
chcp 65001 >nul
setlocal
set "PY=%~dp0..\.venv\Scripts\python.exe"
cd /d "%~dp0.."
if exist "%PY%" goto py_ok
echo ❌ 找不到虚拟环境: %PY%
echo 请先在仓库根目录执行: python3.12 -m venv .venv  并  pip install .
pause
exit /b 1
:py_ok
set "JSON=%TEMP%\yue2_new_song.json"

:edit
echo.
echo [步骤 1/3] 打开记事本, 编辑歌曲信息:
echo   - style  : 语言/曲风/乐器/情绪/BPM + 人声特征(如: 温暖女声, 清亮/沙哑, 气声/强混)
echo   - lyrics : 歌词([Verse]/[Pre-Chorus]/[Chorus]/[Bridge] 段标, 重复段重复写;
echo              可选 [Outro] 尾声段: 写一两句渐弱歌词=人声收尾, 不写=模型器乐淡出)
echo   - 可选改 : cot(full/melody/off) seed(0~2^63 整数, 换值=换一版演唱) id
echo              cfg_scale(0-20, null=自动, 影响对风格的服从度)
echo   ⚠️ 不改占位内容直接生成 = 会生成一首占位示例歌
echo   编辑完请 保存 并 关闭记事本, 回到这里继续
"%PY%" tools\request_builder.py init "%JSON%"
notepad "%JSON%"
"%PY%" tools\request_builder.py validate "%JSON%"
if errorlevel 1 goto bad_edit
goto confirm
:bad_edit
echo.
echo ❌ 内容无效, 即将重新打开记事本修改, 按 Ctrl+C 可退出
pause
goto edit

:confirm
echo.
"%PY%" tools\request_builder.py show "%JSON%"
echo.
set /p CONF=确认生成这首? [y=生成 / n=退出 / 回车或 e=重开编辑]:
if /i "%CONF%"=="y" goto run
if "%CONF%"=="" goto edit
if /i "%CONF%"=="e" goto edit
if /i "%CONF%"=="n" goto quit
echo 请输入 y / n / e 或直接回车
goto confirm
:quit
echo 已退出, 未生成
exit /b 0

:run
echo.
set "TITLE_ARG="
set "ABC_ARG="
set /p TITLE=歌曲标题 TITLE (回车=用 id, 同时决定输出目录名):
if "%TITLE%"=="" goto title_done
set "TITLE_ARG=--title "%TITLE%""
:title_done
if "%TITLE%"=="" goto outdir_plain
"%PY%" tools\request_builder.py outdir "%JSON%" "%TITLE%" > "%TEMP%\yue2_outdir.txt"
goto outdir_got
:outdir_plain
"%PY%" tools\request_builder.py outdir "%JSON%" > "%TEMP%\yue2_outdir.txt"
:outdir_got
set /p OUTDIR=<"%TEMP%\yue2_outdir.txt"
if not exist "%OUTDIR%" goto dir_ok
echo 目录 %OUTDIR% 已存在, 回车换时间戳重试, Ctrl+C 退出
goto run
:dir_ok
echo.
echo 输出目录: %OUTDIR%
set /p ABCFILE=乐谱 ABC 文件路径 (回车=由模型写谱; 需配合 cot=full/melody; 来源: 上次生成目录的 score.abc 或示例 examples\score.abc):
if "%ABCFILE%"=="" goto abc_done
if exist "%ABCFILE%" goto abc_ok
echo ⚠️ 找不到 ABC 文件 "%ABCFILE%", 本次忽略, 由模型写谱
goto abc_done
:abc_ok
set "ABC_ARG=--abc-file "%ABCFILE%""
:abc_done
echo.
echo [步骤 2/3] 开始生成: 音频 + 歌词标签 + 同步 LRC
"%PY%" tools\generate_full.py --output "%OUTDIR%" --request "%JSON%" %ABC_ARG% %TITLE_ARG%
echo.
echo [步骤 3/3] 歌词对齐: 可选 ASR 逐句对齐 (约 3-10 分钟, 覆盖约七成行) 或人工两点校准
set /p ALIGN=运行 ASR 逐句对齐? [a=运行 / 回车=跳过(继续人工两点校准, 也可一路回车全跳)]:
if /i "%ALIGN%"=="a" goto asr_align
set /p VSEC=若走人工: 第一句(verse)开口实测秒数 (回车=接受自动版):
if "%VSEC%"=="" goto cal_done
set /p CSEC=副歌开口实测秒数:
if "%CSEC%"=="" goto cal_done
"%PY%" tools\make_lrc.py "%OUTDIR%" "%VSEC%" "%CSEC%"
echo    ✅ 已按实测锚点重排 lrc; 若中途段落还漂, 把 "歌词=秒数" 发给助手做多点分段
goto cal_done
:asr_align
"%PY%" tools\lyrics_align.py "%OUTDIR%"
:cal_done
echo.
echo ✅ 完成! 试听 %OUTDIR%\audio.flac
pause
exit /b 0