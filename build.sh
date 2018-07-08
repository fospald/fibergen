#!/bin/bash

HOST=$(uname -n | tr " -" "__")
FIBERGEN_DIR=$(cd $(dirname "$0") && pwd)

source /LOCAL/Software/InsightToolkit-4.10.1/setup_env
source /LOCAL/Software/boost-numeric-bindings/setup_env

BUILD_DIR="$FIBERGEN_DIR/build/$HOST"
rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"
cmake -DCMAKE_VERBOSE_MAKEFILE:BOOL=OFF ../..
make -j
RET=$?
cd "$FIBERGEN_DIR"
test $RET -eq 0

