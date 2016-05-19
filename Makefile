CXX=g++
NCXX=nvcc

CUDA_INSTALL_PATH=/usr/local/cuda-7.5
CFLAGS= -I. -I$(CUDA_INSTALL_PATH)/include `pkg-config --cflags opencv`
LDFLAGS= -L$(CUDA_INSTALL_PATH)/lib64 -lcudart `pkg-config --libs opencv`
COMPILE_FLAGS= -mcmodel=large -fPIC -g -Wall
OBJ=obj
BIN=bin
SRC=src

#Uncomment the line below if you dont have CUDA enabled GPU
#EMU=-deviceemu

ifdef EMU
CUDAFLAGS+=-deviceemu
endif

all:
	@mkdir -p $(OBJ)
	@mkdir -p $(BIN)
	$(CXX) $(COMPILE_FLAGS) -c $(SRC)/main.cpp -o $(OBJ)/main.o $(CFLAGS)
	$(NCXX) -c $(SRC)/dehazing.cu -o $(OBJ)/kernels.o $(CUDAFLAGS) 
	$(CXX) $(COMPILE_FLAGS) $(OBJ)/main.o $(OBJ)/kernels.o -o $(BIN)/dehazing $(LDFLAGS)

run:
	$(BIN)/dehazing -o output.png -i img/rock.png

clean:
	rm -f $(OBJ)/* $(BIN)/*
	rm *.png

