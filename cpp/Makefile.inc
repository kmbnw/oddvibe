CC=g++
CFLAGS=-Wall -std=c++11 -g
SHAREDCFLAGS=-shared -fPIC
LIBDIR=lib/
BINDIR=bin/
PROJECT=oddvibe

define cc-command
$(CC) $(CFLAGS)
endef
