@echo off
rem gcc 5.1.0 (tdm64) win10
g++ model.cpp -O3 -std=c++11 -Wall -Wextra -pedantic -DBUILD_LIB -shared -o model.dll
pause