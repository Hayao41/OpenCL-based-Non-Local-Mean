#include <fstream>
#include <iostream>
#include <sstream>
#include <cv.hpp>
#ifdef _APPLE_
#include <OpenCL/cl.h>
#else
#include <CL\cl.h>
#include <CL\cl.hpp>
#endif


void Cleanup(cl_context context, cl_command_queue commandQueue, cl_program program, cl_kernel kernel, cl_mem imageObjects[2], cl_sampler sampler);

void Cleanup(cl_context context, cl_command_queue commandQueue, cl_mem inputImage, cl_mem outputImage);

size_t RoundUp(int groupSize, int globalSize);

double getPSNR(const cv::Mat& I1, const cv::Mat& I2);

unsigned next_multiple(const unsigned x, const unsigned n);

