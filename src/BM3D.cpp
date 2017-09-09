#include "BM3D.h"

BM3DFilter::BM3DFilter(){
	this->test = BlockMatching();
	this->context = NULL;
	this->command_queue = NULL;
	this->denoisedImageBuffer = NULL;
	this->hasImageBuffer = false;
}

BM3DFilter::BM3DFilter(cl_context context,					
					   cl_command_queue command_queue		
					   ){
	this->context = context;
	this->command_queue = command_queue;
}

BM3DFilter::~BM3DFilter(){
	Cleanup(this->context,this->command_queue,this->inputImage,this->outputImage);
	free(this->denoisedImageBuffer);
}

void BM3DFilter::initial(){
	cl_int errNum;
	if (context == NULL){
		std::cout << "OpenCL context is null please create and set cl_context" << std::endl;
		return;
	}
	inputImage = LoadImage(context, image_url, imageWidth, imageHeight);
	if (inputImage == 0){
		std::cerr << "Error loading" << std::endl;
		return;
	}
	cl_image_format format;
	format.image_channel_order = CL_RGBA;
	format.image_channel_data_type = CL_UNORM_INT8;
	this->outputImage = clCreateImage2D(context, CL_MEM_WRITE_ONLY, &format, imageWidth, imageHeight, 0, NULL, &errNum);
	if (errNum != CL_SUCCESS){
		std::cout << "Error creating CL output image object" << std::endl;
	}
	this->denoisedImageBuffer = new char[imageHeight * imageWidth * 4];
}

char* BM3DFilter::getOuptputImage(){
	if (this->hasImageBuffer)
		return this->denoisedImageBuffer;
	return NULL;
}

void BM3DFilter::test_func(){
	cl_mem similar_coords;
	cl_mem block_counts;
	cl_mem cl_out;
	cl_mem cl_test;
	cl_int errNum = 0;
	cl_sampler sampler;
	cl_program program;
	cl_program program2;
	cl_kernel  kernel;
	cl_kernel kernel2;
	std::string cl_kernel_file = "CL_Files/Fast_BlockMatching.cl";
	std::string cl_kernel_file2 = "CL_Files/bm3d_basic_filter.cl";
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
	program2 = CreateProgram(context, device, cl_kernel_file2);
	kernel = clCreateKernel(program, "BlockMatching", NULL);
	kernel2 = clCreateKernel(program2, "bm3d_basic_filter", NULL);

	size_t localWorkSize[2] = { 8, 8 };
	const size_t w = RoundUp(localWorkSize[0], this->imageWidth);
	const size_t h = RoundUp(localWorkSize[1], this->imageHeight);
	const int gx_d = next_multiple((unsigned)this->imageWidth, localWorkSize[0]);
	const int gy_d = next_multiple((unsigned)this->imageHeight, localWorkSize[1]);
	const size_t block_count_size = gx_d * gy_d * sizeof(uchar);
	const size_t similar_coords_size = block_count_size * MAX_BLOCKS * 2 *sizeof(short);
	const size_t test_size = imageWidth * imageHeight * sizeof(float);
	size_t globalWorkSize[2] = {
								(rsize_t)w,
								(rsize_t)h
								};

	clock_t t1, t2;
	short* out = NULL;
	uchar* temp2 = NULL;
	float* test = NULL;
	short* temp = NULL;
	out = (short*)malloc(similar_coords_size);
	temp2 = (uchar*)malloc(block_count_size);
	temp = (short*)malloc(similar_coords_size);
	test = (float*)malloc(test_size);
	for (int i = 0; i < gx_d * gy_d * MAX_BLOCKS * 2; i++){
		temp[i] = SHRT_MIN;
	}

	block_counts = clCreateBuffer(context, CL_MEM_READ_WRITE, block_count_size, NULL, &errNum);
	similar_coords = clCreateBuffer(context, CL_MEM_READ_WRITE, similar_coords_size, NULL, &errNum);
	cl_out = clCreateBuffer(context, CL_MEM_READ_WRITE, similar_coords_size, NULL, &errNum);
	cl_test = clCreateBuffer(context, CL_MEM_READ_WRITE, test_size, NULL, &errNum);

	errNum = clSetKernelArg(kernel, 0, sizeof(cl_mem), &this->inputImage);
	errNum |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &similar_coords);
	errNum |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &block_counts);

	errNum |= clSetKernelArg(kernel2, 0, sizeof(cl_mem), &this->inputImage);
	errNum |= clSetKernelArg(kernel2, 1, sizeof(cl_mem), &this->outputImage);
	errNum |= clSetKernelArg(kernel2, 2, sizeof(cl_mem), &similar_coords);
	errNum |= clSetKernelArg(kernel2, 3, sizeof(cl_mem), &block_counts);
	errNum |= clSetKernelArg(kernel2, 4, sizeof(cl_mem), &cl_out);
	errNum |= clSetKernelArg(kernel2, 5, sizeof(cl_mem), &cl_test);

	if (errNum != CL_SUCCESS){
		std::cout << "Error setting kernel arguments." << std::endl;
	}

	t1 = clock();
	errNum = clEnqueueNDRangeKernel(this->command_queue, kernel, 2, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
	if (errNum != CL_SUCCESS){
		std::cout << "Error running kernel." << std::endl;
	}


	errNum = clEnqueueNDRangeKernel(this->command_queue, kernel2, 2, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
	if (errNum != CL_SUCCESS){
		std::cout << "Error running kernel." << std::endl;
	}
	t2 = clock();
	//clEnqueueReadBuffer(command_queue, cl_out, CL_TRUE, 0, similar_coords_size, out, 0, NULL, NULL);
	//clEnqueueReadBuffer(command_queue, cl_test, CL_TRUE, 0, test_size, test, 0, NULL, NULL);
	clEnqueueReadBuffer(command_queue, block_counts, CL_TRUE, 0, block_count_size, temp2, 0, NULL, NULL);
	//clEnqueueReadBuffer(command_queue, similar_coords, CL_TRUE, 0, similar_coords_size, temp, 0, NULL, NULL);
	size_t origin[3] = { 0, 0, 0 };
	size_t region[3] = { this->imageWidth, this->imageHeight, 1 };
	errNum = clEnqueueReadImage(command_queue,
								this->outputImage,
								CL_TRUE,
								origin,
								region,
								0,
								0,
								denoisedImageBuffer,
								0,
								NULL,
								NULL);
	this->hasImageBuffer = true;
	if (errNum != CL_SUCCESS){
		this->hasImageBuffer = false;
		std::cout << "Error reading result buffer." << std::endl;
	}
	std::cout << "OpenCL Running Time:      " << t2 - t1 << "ms" << std::endl;
	int counter = 0;
	for (int i = 0; i < gx_d * gy_d; i++){
		//printf("---------\n");
		//printf("%d\n", i);
		//printf("---------\n");
		if (temp2[i] == 0){
			//printf("---------\n");
			counter++;
		}
		printf("%d ",temp2[i]);
		//printf("%d",out[i]);
		printf("\n");
		/*if (temp2[i] == 0)
			printf("---------\n");*/
	}
	//printf("\n%d\n",counter);
	//printf("\n");
	//for (int i = 0; i < gx_d * gy_d * MAX_BLOCKS ; i += 2){
	//	/*if (i % 16 == 0){
	//		printf("---------\n");
	//		printf("%d\n" , i / 16);
	//		printf("---------\n");
	//	}*/
	//	//printf("%d %d\n",temp[i],temp[i+1]);
	//	printf("%d %d\n", out[i], out[i + 1]);

	//	if (i == gx_d * gy_d * MAX_BLOCKS - 2){
	//		printf("%d\n",i);
	//	}
	//}

	/*for (int i = 0; i <1000; i++){
		printf("%f\n",test[i]);
	}*/
	errNum |= clReleaseKernel(kernel);
	errNum |= clReleaseProgram(program);
	errNum |= clReleaseKernel(kernel2);
	errNum |= clReleaseProgram(program2);
	errNum |= clReleaseMemObject(similar_coords);
	errNum |= clReleaseMemObject(block_counts);
	errNum |= clReleaseMemObject(cl_out);
	errNum |= clReleaseSampler(sampler);
	free(temp2);
	free(temp);
	free(out);
	free(test);
}

void BM3DFilter::test_func2(){

	cl_program program;
	cl_kernel dist_kernel;
	cl_kernel basic_kernel;
	cl_int errNum;

	const size_t ls[2] = { 16, 8 };
	const int gx_d = next_multiple((unsigned)ceil(this->imageWidth / (double)STEP_SIZE), ls[0]);
	const int gy_d = next_multiple((unsigned)ceil(this->imageHeight / (double)STEP_SIZE), ls[1]);
	const int tot_items_d = gx_d * gy_d;

	const size_t gx = next_multiple((unsigned)ceil(this->imageWidth / (double)SPLIT_SIZE_X), ls[0]);
	const size_t gy = next_multiple((unsigned)ceil(this->imageHeight / (double)SPLIT_SIZE_Y), ls[1]);
	const size_t tot_items = gx * gy;

	const cl_int hard_threshold = D_THRESHOLD_1 * BLOCK_SIZE_SQ;
	const cl_int wiener_threshold = D_THRESHOLD_2 * BLOCK_SIZE_SQ;
	const cl_int max_block_count_1 = MAX_BLOCK_COUNT_1;
	const cl_int max_block_count_2 = MAX_BLOCK_COUNT_2;
	const cl_int window_step_size_1 = WINDOW_STEP_SIZE_1;
	const cl_int window_step_size_2 = WINDOW_STEP_SIZE_2;
	const cl_int WIDTH = this->imageWidth;
	const cl_int HEIGHT = this->imageHeight;

	std::string cl_kernel_file = "CL_Files/bm3d.cl";

	program = CreateProgram(context, device, cl_kernel_file);

	dist_kernel = clCreateKernel(program, "calc_distances", NULL);
	//basic_kernel = clCreateKernel(program, "bm3d_basic_filter", NULL);

	const size_t similar_coords_size = MAX_BLOCK_COUNT_2 * tot_items_d * sizeof(cl_short)* 2;

	cl_mem similar_coords_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, similar_coords_size, NULL, &errNum);

	const size_t block_counts_size = tot_items_d * sizeof(cl_uchar);
	cl_mem block_counts_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, block_counts_size, NULL, &errNum);


	errNum = clSetKernelArg(dist_kernel, 0, sizeof(cl_mem), &this->inputImage);
	errNum |= clSetKernelArg(dist_kernel, 1, sizeof(cl_mem), &similar_coords_buffer);
	errNum |= clSetKernelArg(dist_kernel, 2, sizeof(cl_mem), &block_counts_buffer);
	errNum |= clSetKernelArg(dist_kernel, 3, sizeof(cl_int), &hard_threshold);
	errNum |= clSetKernelArg(dist_kernel, 4, sizeof(cl_int), &max_block_count_1);
	errNum |= clSetKernelArg(dist_kernel, 5, sizeof(cl_int), &window_step_size_1);

	if (errNum != CL_SUCCESS){
		std::cout << "Error setting kernel arguments." << std::endl;
	}


	//errNum = clSetKernelArg(basic_kernel, 0, sizeof(cl_mem), &this->outputImage);
	//errNum |= clSetKernelArg(basic_kernel, 1, sizeof(cl_mem), &this->outputImage);
	//errNum |= clSetKernelArg(basic_kernel, 2, sizeof(cl_mem), &similar_coords_buffer);
	//errNum |= clSetKernelArg(basic_kernel, 3, sizeof(cl_mem), &block_counts_buffer);
	//errNum |= clSetKernelArg(basic_kernel, 4, sizeof(cl_int), &gx_d);
	//errNum |= clSetKernelArg(basic_kernel, 5, sizeof(cl_int), &tot_items_d);
	//errNum |= clSetKernelArg(basic_kernel, 6, sizeof(cl_int), &WIDTH);
	//errNum |= clSetKernelArg(basic_kernel, 7, sizeof(cl_int), &HEIGHT);

	//if (errNum != CL_SUCCESS){
	//	std::cout << "Error setting kernel arguments." << std::endl;
	//}

	//clock_t t1, t2;

	for (int j = 0; j < 1; j++) {
		for (int i = 0; i < 1; i++) {
			size_t gs_d[2] = { gx_d, gy_d };
			size_t gs[2] = { gx, gy };
			size_t offset[2] = { i*gx, j*gy };
			assert(ls[0] * ls[1] <= maxWG);
			assert(!(gs[0] % ls[0]));
			assert(!(gs[1] % ls[1]));

			//t1 = clock();

			errNum = clEnqueueNDRangeKernel(this->command_queue, dist_kernel, 2, offset, gs_d, ls, 0, NULL, NULL);
			if (errNum != CL_SUCCESS){
				std::cout << "Error running kernel." << std::endl;
			}
			/*errNum |= clEnqueueNDRangeKernel(this->command_queue, basic_kernel, 2, offset, gs, ls, 0, NULL, NULL);
			if (errNum != CL_SUCCESS){
				std::cout << "Error running kernel." << std::endl;
			}*/

			//t2 = clock();
			/*size_t origin[3] = { 0, 0, 0 };
			size_t region[3] = { this->imageWidth, this->imageHeight, 1 };
			errNum = clEnqueueReadImage(command_queue,
										this->outputImage,
										CL_TRUE,
										origin,
										region,
										0,
										0,
										denoisedImageBuffer,
										0,
										NULL,
										NULL);
			this->hasImageBuffer = true;
			if (errNum != CL_SUCCESS){
				this->hasImageBuffer = false;
				std::cout << "Error reading result buffer." << std::endl;
			}
			std::cout << "OpenCL Running Time:      " << t2 - t1 << "ms" << std::endl;*/
		}
	}

}