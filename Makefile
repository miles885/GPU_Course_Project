BIN = assignment.exe

# Compiler flags
CC = g++
CC_FLAGS = -std=c++11

# Sources
SOURCES = Assignment.cpp

# Objects
OBJS = $(patsubst %.cpp, %.o, $(SOURCES))

#Includes
INC = -I/usr/local/cuda/include

# Libraries
LIB_DIRS = -L/usr/local/cuda/lib64

LIBS = -lOpenCL

.PHONY: all

all: $(OBJS)
	$(CC) -o $(BIN) $(OBJS) $(INC) $(LIB_DIRS) $(LIBS)

%.o: %.cpp
	$(CC) -c $(CC_FLAGS) $(INC) -o $@ $<

clean:
	rm -f $(BIN) $(OBJS)
