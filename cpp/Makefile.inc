CC=g++
CFLAGS=-Wall -pedantic -std=c++11 -O3 -lpthread
SHAREDCFLAGS=-shared -fPIC
LIBDIR=lib/
BINDIR=bin/
PROJECT=oddvibe

define cc-command
$(CC) $(CFLAGS)
endef
