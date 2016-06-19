CXX=g++ -std=c++11
NCXX=nvcc

CUDA_INSTALL_PATH=/usr/local/cuda-7.5
CFLAGS=-I. -I$(CUDA_INSTALL_PATH)/include `pkg-config --cflags opencv` 
LDFLAGS=-L$(CUDA_INSTALL_PATH)/lib64 -lcudart `pkg-config --libs opencv` -lboost_system -lboost_timer -lboost_program_options
COMPILE_FLAGS= -mcmodel=large -fPIC -g -Wall
OBJ=obj
BIN=bin
SRC=src

CUDAFLAGS=-lcusolver -lcusparse

ifdef EMU
CUDAFLAGS+=-deviceemu 
endif

all:
	@mkdir -p $(OBJ)
	@mkdir -p $(BIN)
	$(CXX) $(COMPILE_FLAGS) -c $(SRC)/HazeRemover.cpp -o $(OBJ)/HazeRemover.o $(CFLAGS)
	$(NCXX) -c $(SRC)/dehazing.cu -o $(OBJ)/main_kernel.o $(CUDAFLAGS) 
	$(NCXX) -c $(SRC)/guidedfilter.cu -o $(OBJ)/guided_kernel.o $(CUDAFLAGS) 
	$(NCXX) -c $(SRC)/softmatting.cu -o $(OBJ)/matting_kernel.o $(CUDAFLAGS) 
	$(CXX) $(COMPILE_FLAGS) $(SRC)/main.cpp $(OBJ)/main_kernel.o $(OBJ)/guided_kernel.o $(OBJ)/matting_kernel.o $(OBJ)/HazeRemover.o -o $(BIN)/dehazing $(CFLAGS) $(LDFLAGS) $(CUDAFLAGS) 

run:
	#$(BIN)/dehazing -i video/cross.avi
	#$(BIN)/dehazing -i video/hazeroad.avi
	$(BIN)/dehazing -i video/riverside.avi
	#$(BIN)/dehazing -i img/city2.jpg
	#$(BIN)/dehazing -i img/city.jpeg
	#$(BIN)/dehazing -i img/forest.jpg
	#$(BIN)/dehazing -i img/forest_small.jpg
	#$(BIN)/dehazing -i img/forest_tiny.jpg
	#$(BIN)/dehazing -i img/rock.png
	#$(BIN)/dehazing -i img/rock_small.png

clean:
	rm -f $(OBJ)/* $(BIN)/*
	rm *.png

