#include <stdio.h> 

// Used to verify that the data is serialized correctly device-side
#define TEST_VALUE -12345 

class ConstructorExample {
public:
	int var;
	
	ConstructorExample(int var) {
		this->var = var; 
	}
	
	// Since this class has a copy constructor, you cannot pass it 
	// to a device-side kernel launch.
	__host__ __device__ ConstructorExample(const ConstructorExample& other) {
		var = other.var; 
	}
};

#if defined (DEVICE_COPY_CONSTRUCTOR_ERROR)

// This is what we would normally do, but ConstructorExample has a copy
// constructor so this will yield an error. 
__global__ void kernel_sub(ConstructorExample obj) {
	printf("Sub-kernel invoked: %d\n", obj.var); 
}

#else 

// Instead, pass a byte array containing the data of the class we want
// to pass. 
__global__ void kernel_sub(char *obj_serialized) {
	ConstructorExample obj = *(static_cast<ConstructorExample*>((void*)obj_serialized));
	
	printf("Sub-kernel invoked\n");
	if (obj.var != TEST_VALUE) {
		printf("Error: passed object not serialized correctly (%d)\n", obj.var); 
	}
}

#endif 

__global__ void kernel_base(ConstructorExample obj) {
	printf("Base-kernel invoked\n"); 
	
#if defined(DEVICE_COPY_CONSTRUCTOR_ERROR)
	
	// Since obj has a copy constructor, this will yield an error. 
	kernel_sub<<<1,1>>>(obj); 

#else
	
	// Copy the data to a byte array and pass that instead. 
	char *serialized = (char*)malloc(sizeof(obj)); 
	memcpy(serialized, &obj, sizeof(obj)); 
	kernel_sub<<<1,1>>>(serialized); 
	free(serialized); 
	
#endif 

}

int main(void) {
	ConstructorExample obj1(TEST_VALUE); 
	ConstructorExample obj2 = obj1; // Copy constructor invoked
	
	if (obj1.var != obj2.var) {
		printf("Error: copy-constructor not being invoked (%d != %d)\n", obj1.var, obj2.var); 
		exit(1); 
	}
	
	printf("Starting kernel example...\n"); 
	kernel_base<<<1,1>>>(obj1);
	
	cudaError_t launch_error = cudaPeekAtLastError(); 
	cudaError_t synchronize_error = cudaDeviceSynchronize(); 
	if (launch_error != cudaSuccess || synchronize_error != cudaSuccess) {
		printf("Error in kernel: %s\n", cudaGetErrorString(launch_error != cudaSuccess ? launch_error : synchronize_error)); 
		exit(1); 
	}
	
	printf("Finished!\n"); 
}