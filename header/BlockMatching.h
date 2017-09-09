#include "Utils.h"
#include "getDevice.h"
#include <time.h>
#include <iostream>
#include <string>

#ifndef __BLOCKMATCHING_H__
#define __BLOCKMATCHING_H__

class BlockMatching{
private:
public:
	BlockMatching(){};
	~BlockMatching(){}

	/* non local mean for testing framework*/
	void NonLocalMean(	cl_context context,
						cl_command_queue commandQueue,
						cl_mem* inputImage,
						cl_mem* outputImage,
						cl_device_id device,
						int width,
						int height,
						char*  denoisedImageBuffer
						);
};
#endif