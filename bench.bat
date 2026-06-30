@echo off
setlocal

cargo build --release
if errorlevel 1 exit /b %errorlevel%

set SABLE_SIMD_BACKEND=avx512
.\target\release\sable-engine.exe bench

set SABLE_SIMD_BACKEND=avx2
.\target\release\sable-engine.exe bench

set SABLE_SIMD_BACKEND=scalar
.\target\release\sable-engine.exe bench

set SABLE_SIMD_BACKEND=
.\target\release\sable-engine.exe bench
