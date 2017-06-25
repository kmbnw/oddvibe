include ../Makefile.inc

TARGET := ../$(LIBDIR)lib$(PROJECT).so
RCPP_SRC = ../../src/

SRC_FILES := $(wildcard $(RCPP_SRC)*.cpp)
SRC_FILES := $(filter-out $(RCPP_SRC)rcpp_oddvibe.cpp, $(SRC_FILES))
SRC_FILES := $(filter-out $(RCPP_SRC)RcppExports.cpp, $(SRC_FILES))
HEADER_FILES := $(wildcard $(RCPP_SRC)*.h)

all: $(TARGET)

$(TARGET): $(SRC_FILES) $(HEADER_FILES)
	mkdir -p ../$(LIBDIR)
	$(cc-command) $(SHAREDCFLAGS) -o $(TARGET) $(SRC_FILES)

clean:
	$(RM) $(TARGET)
