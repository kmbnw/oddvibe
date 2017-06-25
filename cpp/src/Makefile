include ../Makefile.inc

TARGET=../$(LIBDIR)lib$(PROJECT).so

all: $(TARGET)

$(TARGET): *.cpp *.h
	mkdir -p ../$(LIBDIR)
	$(cc-command) $(SHAREDCFLAGS) -o $(TARGET) *.cpp

clean:
	$(RM) $(TARGET)
