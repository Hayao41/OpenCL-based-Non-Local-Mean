#include "BlockMatching.h"

void BlockMatching::NonLocalMean(cl_context context,
								cl_command_queue commandQueue,
								cl_mem* inputImage,
								cl_mem* outputImage,
								cl_device_id device,
								int width,
								int height,
								char*  denoisedImageBuffer
								){
	cl_int errNum = 0;
	float nlmNoise = sqrt(4.0f);
	float nosie = 1.0f / (nlmNoise * nlmNoise);
	float lerpC = 0.2f;
	cl_mem cl_width = NULL;
	cl_mem cl_height = NULL;
	cl_mem cl_lerpC = NULL;
	cl_mem cl_noise = NULL;
	cl_sampler sampler;
	cl_program program;
	cl_kernel  kernel;
	std::string cl_kernel_file = "CL_Files/NonLocalMean.cl";
	sampler = clCreateSampler(context,
		CL_FALSE, // Non-normalized coordinates  
		CL_ADDRESS_CLAMP_TO_EDGE,
		CL_FILTER_NEAREST,
		&errNum);

	if (errNum != CL_SUCCESS)
	{
		std::cerr << "Error creating CL sampler object." << std::endl;
	}

	program = CreateProgram(context, device, cl_kernel_file);

	kernel = clCreateKernel(program, "NLMFiltering", NULL);

	cl_width = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_mem), NULL, &errNum);
	cl_height = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_mem), NULL, &errNum);
	cl_noise = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_mem), NULL, &errNum);
	cl_lerpC = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_mem), NULL, &errNum);

	errNum = clEnqueueWriteBuffer(commandQueue, cl_width, CL_TRUE, 0, sizeof(int), (void*)&width, 0, NULL, NULL);
	errNum |= clEnqueueWriteBuffer(commandQueue, cl_height, CL_TRUE, 0, sizeof(int), (void*)&height, 0, NULL, NULL);
	errNum |= clEnqueueWriteBuffer(commandQueue, cl_noise, CL_TRUE, 0, sizeof(float), (void*)&nosie, 0, NULL, NULL);
	errNum |= clEnqueueWriteBuffer(commandQueue, cl_lerpC, CL_TRUE, 0, sizeof(float), (void*)&lerpC, 0, NULL, NULL);

	size_t localWorkSize[2] = { 8, 8 };
	size_t globalWorkSize[2] = {
		(rsize_t)RoundUp(localWorkSize[0], width),
		(rsize_t)RoundUp(localWorkSize[1], height)
	};
	clock_t t1, t2;
	t1 = clock();
	errNum |= clSetKernelArg(kernel, 0, sizeof(cl_mem), inputImage);
	errNum |= clSetKernelArg(kernel, 1, sizeof(cl_mem), outputImage);
	errNum |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &cl_width);
	errNum |= clSetKernelArg(kernel, 3, sizeof(cl_mem), &cl_height);
	errNum |= clSetKernelArg(kernel, 4, sizeof(cl_mem), &cl_noise);
	errNum |= clSetKernelArg(kernel, 5, sizeof(cl_mem), &cl_lerpC);
	errNum = clEnqueueNDRangeKernel(commandQueue, kernel, 2, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
	size_t origin[3] = { 0, 0, 0 };
	size_t region[3] = { width, height, 1 };
	errNum = clEnqueueReadImage(commandQueue,
								*outputImage,
								CL_TRUE,
								origin,
								region,
								0,
								0,
								denoisedImageBuffer,
								0,
								NULL,
								NULL);
	if (errNum != CL_SUCCESS){
		std::cout << "Error reading result buffer." << std::endl;
		Cleanup(context, commandQueue, *inputImage, *outputImage);
	}
	t2 = clock();
	std::cout << "OpenCL Running Time:      " << t2 - t1 << "ms" << std::endl;

}
