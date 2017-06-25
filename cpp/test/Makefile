include ../Makefile.inc

TARGET=$(BINDIR)$(PROJECT)_test

all: $(TARGET)

$(TARGET): *.cpp *.h
	mkdir -p $(BINDIR)
	$(cc-command) -I ../src -L ../$(LIBDIR) -o $(TARGET) *.cpp -l$(PROJECT) -lcppunit

clean:
	$(RM) $(TARGET)

tests: all
	LD_LIBRARY_PATH=../$(LIBDIR) ./$(TARGET)

debug_tests: all
	LD_LIBRARY_PATH=../$(LIBDIR) ddd ./$(TARGET)
