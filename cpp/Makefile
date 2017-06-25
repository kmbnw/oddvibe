include Makefile.inc

all:
	cd main; make all

clean:
	cd main; make clean
	cd test; make clean

tests: all
	cd test; make tests

debug_tests: all
	cd test; make debug_tests
