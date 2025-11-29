include config/make.conf

all: config/make.conf
	cd integrations; make all
	cd src; make all

clean: config/make.conf
	cd src; make clean

config/make.conf:
	@echo "You should configure before building"
	@echo "Use: './configure --help' for options"
	@exit 1

install: all
	mkdir -p $(PREFIX)/{include,lib,pylib,bin}
	cd include; install -m 664 *.h $(PREFIX)/include
	@# set nullglob so that on linux it survives no dylib files, and on mac no .so files
	cd lib; shopt -s nullglob; install -m 664 *.a *.{so,dylib} $(PREFIX)/lib
	cd pylib; install -m 664 *.py $(PREFIX)/pylib
	cd bin; install -m 775 * $(PREFIX)/bin
