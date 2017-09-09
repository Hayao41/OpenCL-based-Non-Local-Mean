#include "Utils.h"

void Cleanup(cl_context context, cl_command_queue commandQueue, cl_program program, cl_kernel kernel, cl_mem imageObjects[2], cl_sampler sampler)
{
	for (int i = 0; i < 2; i++)
	{
		if (imageObjects[i] != 0)
			clReleaseMemObject(imageObjects[i]);
	}
	if (commandQueue != 0)
		clReleaseCommandQueue(commandQueue);

	if (kernel != 0)
		clReleaseKernel(kernel);

	if (program != 0)
		clReleaseProgram(program);

	if (sampler != 0)
		clReleaseSampler(sampler);

	if (context != 0)
		clReleaseContext(context);

}

void Cleanup(cl_context context, cl_command_queue commandQueue , cl_mem inputImage, cl_mem outputImage)
{
	if (commandQueue != 0)
		clReleaseCommandQueue(commandQueue);

	if (context != 0)
		clReleaseContext(context);

	if (inputImage != 0)
		clReleaseMemObject(inputImage);

	if (outputImage != 0)
		clReleaseMemObject(outputImage);

}

size_t RoundUp(int groupSize, int globalSize)
{
	int r = globalSize % groupSize;
	if (r == 0)
	{
		return globalSize;
	}
	else
	{
		return globalSize + groupSize - r;
	}
}

double getPSNR(const cv::Mat& I1, const cv::Mat& I2)
{
	cv::Mat s1;
	absdiff(I1, I2, s1);       // |I1 - I2|
	s1.convertTo(s1, CV_32F);  // cannot make a square on 8 bits
	s1 = s1.mul(s1);           // |I1 - I2|^2

	cv::Scalar s = sum(s1);         // sum elements per channel

	double sse = s.val[0] + s.val[1] + s.val[2]; // sum channels

	if (sse <= 1e-10) // for small values return zero
		return 0;
	else
	{
		double  mse = sse / (double)(I1.channels() * I1.total());
		double psnr = 10.0*log10((255 * 255) / mse);
		return psnr;
	}
}

unsigned next_multiple(const unsigned x, const unsigned n) {
	return (x - (x % n)) / n;
	//return (x + n - 1) & ~(n - 1);
}