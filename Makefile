BIN = project.exe

# Compiler flags
CC = nvcc
CC_FLAGS = -std=c++11

# Sources
SOURCES = src/EdgeDetection.cu \
          src/ImageProc.cu \
          src/ImageUtils.cu

# Objects
OBJS = $(patsubst %.cpp, %.o, $(SOURCES))
OBJS := $(patsubst %.cu, %.o, $(SOURCES))

#Includes
INC = -I/usr/include \
      -I./inc \
      -I./src

# Libraries
LIB_DIRS = -L/usr/local/cuda/lib64 \
           -L./lib

LIBS = -lcudart \
       -lfreeimage

.PHONY: all

all: $(OBJS)
	$(CC) $(INC) $(LIB_DIRS) $(LIBS) -o $(BIN) $(OBJS)

%.o: %.cpp
	$(CC) -c $(CC_FLAGS) $(INC) -o $@ $<

%.o: %.cu
	$(CC) -c $(CC_FLAGS) $(INC) -o $@ $<

clean:
	rm -f $(BIN) $(OBJS)
