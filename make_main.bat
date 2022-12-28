@echo off
rem gcc 5.1.0 (tdm64) win10
g++ main.cpp -O3 -std=c++11 -Wall -Wextra -pedantic -L./ -lmodel -o main.exe
pause