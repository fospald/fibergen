#!/bin/bash

HOST=$(uname -n | tr " -" "__")
FIBERGEN_DIR=$(cd $(dirname "$0") && pwd)

BUILD_DIR="$FIBERGEN_DIR/build/$HOST"
mkdir -p "$BUILD_DIR" || exit
cd "$BUILD_DIR" || exit
cmake -DCMAKE_VERBOSE_MAKEFILE:BOOL=OFF "$@" ../.. || exit
make -j || exit

echo ""
echo "Build successful!"
echo "Please update your environmet variables, i.e.:"
echo "export PATH=\$PATH:$FIBERGEN_DIR/bin"
echo "export PYTHONPATH=\$PYTHONPATH:$FIBERGEN_DIR/lib"

