CXX=g++ -std=c++11
NCXX=nvcc

CUDA_INSTALL_PATH=/usr/local/cuda-7.5
CFLAGS= -I. -I$(CUDA_INSTALL_PATH)/include `pkg-config --cflags opencv`
LDFLAGS= -L$(CUDA_INSTALL_PATH)/lib64 -lcudart `pkg-config --libs opencv` -lboost_system -lboost_timer -lboost_program_options
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
	$(CXX) $(COMPILE_FLAGS) -c $(SRC)/HazeRemover.cpp -o $(OBJ)/HazeRemover.o $(CFLAGS)
	$(NCXX) -c $(SRC)/dehazing.cu -o $(OBJ)/kernels.o $(CUDAFLAGS) 
	$(CXX) $(COMPILE_FLAGS) $(SRC)/main.cpp $(OBJ)/kernels.o $(OBJ)/HazeRemover.o -o $(BIN)/dehazing $(CFLAGS) $(LDFLAGS) 

run:
	$(BIN)/dehazing -o output.png -i img/rock.png

clean:
	rm -f $(OBJ)/* $(BIN)/*
	rm *.png

