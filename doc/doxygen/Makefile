
DOXYFILE:=Doxyfile

DOXYGEN:=doxygen

all: doxygen images

images:
	cp screenshot*.png html

doxygen: $(DOXYFILE)
	$(DOXYGEN) $<

clean:
	rm -rf html latex
