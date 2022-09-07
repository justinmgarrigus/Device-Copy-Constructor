FLAGS = 

all: 
	nvcc -arch=sm_70 -dc error.cu -o error.o -lcudadevrt $(FLAGS)
	nvcc -arch=sm_70 -dlink error.o -o link.o -lcudadevrt $(FLAGS)
	g++ error.o link.o -lcudadevrt -lcudart -L/usr/local/cuda/lib64 $(FLAGS) 

fixed: 
	make FLAGS=""

error: 
	make FLAGS="-DDEVICE_COPY_CONSTRUCTOR_ERROR"