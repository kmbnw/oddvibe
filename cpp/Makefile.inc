CC=g++
CFLAGS=-Wall -std=c++14 -O3
SHAREDCFLAGS=-shared -fPIC
LIBDIR=../inst/lib/
BINDIR=../inst/bin/
PROJECT=oddvibe

define cc-command
$(CC) $(CFLAGS)
endef
