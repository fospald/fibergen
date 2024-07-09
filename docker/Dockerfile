FROM ubuntu:20.04

WORKDIR /code

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
	cmake \
	build-essential \
	git \
	python2 \
	python3 \
	python3-dev \
	python3-numpy-dev \
	libpng-dev \
	libboost-all-dev \
	libfftw3-dev \
	libopenblas-dev \
	python3-scipy \
	python3-pyqt5 \
	python3-pyqt5.qtwebengine \
	python3-matplotlib \
	fonts-oflb-asana-math \
	vim

RUN git clone https://github.com/inducer/boost-numeric-bindings/
RUN cd /code/boost-numeric-bindings/ && python2 ./configure --prefix=/code/boost-numeric-bindings/ && make install
ENV BOOSTNUMERICBINDINGS_DIR=/code/boost-numeric-bindings/include/boost-numeric-bindings/
ENV PATH="$PATH:/code/fibergen/bin"
ENV PYTHONPATH="$PYTHONPATH:/code/fibergen/lib"

ARG DEBIAN_FRONTEND=dialog

