#include <stdio.h>
#include <CL/opencl.h>

const char *test_kernel = ""
"__kernel void vector_add(__global const int *A, __global const int *B, __global const int *C) {"
"	int i get_global_id(0);"
""
"	C[i] = A[i] + B[i];"
"}";

int main() {
	cl_platform_id platform_id = NULL;
	cl_device_id device_id = NULL;
	cl_uint ret_num_devices;
	cl_uint ret_num_platforms;
	cl_int ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
	ret = clGetDeviceIDs( platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id, &ret_num_devices);

	int *A = malloc(100*sizeof(int));
	int *B = malloc(100*sizeof(int));
	int *C = malloc(100*sizeof(int));

	for (int i = 0; i < 100; i++) {
		A[i] = i;
		B[i] = i;
		C[i] = 0;
	}

	 // Create an OpenCL context
	cl_context context = clCreateContext( NULL, 1, &device_id, NULL, NULL, &ret);

	// Create a command queue
	cl_command_queue command_queue = clCreateCommandQueue(context, device_id, 0, &ret);

	// Create memory buffers on the device for each vector
	cl_mem a_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY,
			100 * sizeof(int), NULL, &ret);
	cl_mem b_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY,
			100 * sizeof(int), NULL, &ret);
	cl_mem c_mem_obj = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
			100 * sizeof(int), NULL, &ret);

	// Copy the lists A and B to their respective memory buffers
	ret = clEnqueueWriteBuffer(command_queue, a_mem_obj, CL_TRUE, 0,
			100 * sizeof(int), A, 0, NULL, NULL);
	ret = clEnqueueWriteBuffer(command_queue, b_mem_obj, CL_TRUE, 0,
			100 * sizeof(int), B, 0, NULL, NULL);

	// Create a program from the kernel source
	cl_program program = clCreateProgramWithSource(context, 1,
			(const char **)&test_kernel, NULL, &ret);

	// Build the program
	ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);

	// Create the OpenCL kernel
	cl_kernel kernel = clCreateKernel(program, "vector_add", &ret);

	// Set the arguments of the kernel
	ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&a_mem_obj);
	ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&b_mem_obj);
	ret = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&c_mem_obj);

	size_t global_item_size = 100;
	size_t local_item_size = 100;
	ret = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, &global_item_size, &local_item_size, 0, NULL, NULL);

	if (ret != CL_SUCCESS) {
		fprintf(stderr, "something went wrong %d\n", ret);
	}

	ret = clEnqueueReadBuffer(command_queue, c_mem_obj, CL_TRUE, 0, 100*sizeof(int), C, 0, NULL, NULL);

	int sum = 0;
	for (int i = 0; i < 100; i++) {
		sum += C[i];
	}

	printf("sum: %d (should be) %d\n", sum, 2*100*(100-1)/2);

	ret = clFlush(command_queue);
	ret = clFinish(command_queue);
	ret = clReleaseKernel(kernel);
	ret = clReleaseProgram(program);
	ret = clReleaseMemObject(a_mem_obj);
	ret = clReleaseMemObject(b_mem_obj);
	ret = clReleaseMemObject(c_mem_obj);
	ret = clReleaseCommandQueue(command_queue);
	ret = clReleaseContext(context);
}
