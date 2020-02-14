#!/bin/sh
clang -g `llvm-config --cflags` test.c -c
clang++ test.o `llvm-config --cxxflags --ldflags --libs --system-libs all` -o test
