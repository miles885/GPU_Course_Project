BIN = assignment.exe

# Compiler flags
CC = nvcc
CC_FLAGS = -std=c++11

# Sources
SOURCES = assignment.cu \
          npp_nvgraph.cu \
          thrust.cu

# Objects
OBJS = $(patsubst %.cpp, %.o, $(SOURCES))
OBJS := $(patsubst %.cu, %.o, $(SOURCES))

#Includes
INC = -I/usr/include \
      -I../common/inc \
      -I./common/FreeImage/include \
      -I./common/UtilNPP

# Libraries
LIB_DIRS = -L/usr/local/cuda/lib64 \
           -L./common/FreeImage/lib/linux/x86_64

LIBS = -lcudart \
       -lnppc \
       -lnppisu \
       -lnppial \
       -lnvgraph \
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
