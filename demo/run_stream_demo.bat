@echo off
setlocal

rem Run from this folder so WSL can use the current directory under /mnt.
pushd "%~dp0"

rem With no arguments, run the bundled sample-video smoke test.
if "%~1"=="" (
    set "STREAM_ARGS=--video docs/source/assets/viewer_garden_480p.mp4 --steps 100 --no-difix --no-view --no-lbfgs"
) else (
    set "STREAM_ARGS=%*"
)

wsl.exe -d Ubuntu -- bash -lc "source /home/chron/miniforge3/etc/profile.d/conda.sh && conda activate /home/chron/3dgrut && export HF_HOME=/home/chron/hf_cache HYPERSPLAT_ACEZERO_PYTHON=/home/chron/.conda_envs/ace0/bin/python CUDA_HOME=/home/chron/3dgrut CPATH=/home/chron/3dgrut/targets/x86_64-linux/include CC=/home/chron/3dgrut/bin/x86_64-conda-linux-gnu-gcc CXX=/home/chron/3dgrut/bin/x86_64-conda-linux-gnu-g++ && python3 -c \"from pathlib import Path; p=Path('run_stream_demo.sh'); Path('.run_stream_demo.wsl.sh').write_bytes(p.read_bytes().replace(bytes([13]), b''))\" && trap 'rm -f .run_stream_demo.wsl.sh' EXIT && bash \"$PWD/.run_stream_demo.wsl.sh\" %STREAM_ARGS%"
set "RESULT=%ERRORLEVEL%"

popd
exit /b %RESULT%
