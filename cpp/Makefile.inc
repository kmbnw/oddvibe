CC=g++
CFLAGS=-Wall -std=c++11 -O3
SHAREDCFLAGS=-shared -fPIC
LIBDIR=lib/
BINDIR=bin/
PROJECT=oddvibe

define cc-command
$(CC) $(CFLAGS)
endef
