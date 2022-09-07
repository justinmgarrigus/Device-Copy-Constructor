#include <stdio.h> 

class ConstructorExample {
public:
	int var;
	
	ConstructorExample(int var) {
		this->var = var; 
	}
	
	__host__ __device__ ConstructorExample(const ConstructorExample& other) {
		var = other.var; 
		printf("Copy constructor invoked\n"); 
	}
};

#if defined (DEVICE_COPY_CONSTRUCTOR_ERROR)

__global__ void kernel_sub(ConstructorExample obj) {
	printf("Sub-kernel invoked: %d\n", obj.var); 
}

#else 

__global__ void kernel_sub(char obj_serialized[sizeof(ConstructorExample)]) {
	ConstructorExample obj = *(static_cast<ConstructorExample*>((void*)obj_serialized));
	
	printf("Sub-kernel invoked: %d\n", obj.var); 
}

#endif 

__global__ void kernel_base(ConstructorExample obj) {
	printf("Base-kernel invoked: %d\n", obj.var); 
	
#if defined(DEVICE_COPY_CONSTRUCTOR_ERROR)

	kernel_sub<<<1,1>>>(obj); 

#else
	
	char *serialized = (char*)malloc(sizeof(obj)); 
	memcpy(serialized, &obj, sizeof(obj)); 
	kernel_sub<<<1,1>>>(serialized); 
	free(serialized); 
	
#endif 

}

int main(void) {
	ConstructorExample obj1(-12345); 
	printf("Obj1: %d\n", obj1.var); 
	
	ConstructorExample obj2 = obj1; 
	printf("Obj2: %d\n", obj2.var); 
	
	printf("Starting kernel example...\n"); 
	kernel_base<<<1,1>>>(obj1);
	
	cudaError_t launch_error = cudaPeekAtLastError(); 
	cudaError_t synchronize_error = cudaDeviceSynchronize(); 
	if (launch_error != cudaSuccess || synchronize_error != cudaSuccess) {
		fprintf(stderr, "Error in kernel: %s\n", cudaGetErrorString(launch_error != cudaSuccess ? launch_error : synchronize_error)); 
		exit(1); 
	}
	
	printf("Finished!\n"); 
}