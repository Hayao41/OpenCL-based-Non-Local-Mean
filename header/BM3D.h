#include <iostream>
#include <vector>
#include <string>
#include <sstream>

#include <cstdio>
#include <cmath>
#include <cassert>
#include <cstring>

#include "LoadImage.h"
#include "Utils.h"
#include "config.h"
#include "BlockMatching.h"

/**
*@Author      : Joel Chen
*@Time        : 2017/5/8
*@Description : BM3D Filter definition
*               this file define the BM3D filter which implements the BM3D denoising
*               algorithm
*/
class BM3DFilter{
private:
	BlockMatching test;                    
	cl_mem inputImage;					//input image global memory
	cl_mem outputImage;					//denosied image global memory
	cl_context context;					//cl context
	cl_command_queue command_queue;		//cl command queue
	cl_device_id device;				//deivce id
	std::string image_url;				//the input image url 
	char*  denoisedImageBuffer;			//denoised image buffer
	int imageWidth;						//width of input image
	int imageHeight;					//height of input image
	bool hasImageBuffer;				//is the imageBuffer filled by denosied image

	/**
	*initialize filter after set the waitting for denoised image
	*@return void
	*/
	void initial();

public:

	/**
	*implicit constructor
	*/
	BM3DFilter();

	/**
	*explicit constructor
	*/
	explicit BM3DFilter(cl_context context,					//opencl context
						cl_command_queue command_queue		//opencl command queue
						);

	/**
	*deconstructor
	*/
	~BM3DFilter();

	/*several setter function */

	void setUrl(std::string url){
		this->image_url = url;
		this->initial();
	}

	void setContext(cl_context context){
		this->context = context;
	}

	void setCommandQueue(cl_command_queue command_queue){
		this->command_queue = command_queue;
	}

	void setDevice(cl_device_id device){
		this->device = device;
	}

	/**
	*retrun the denoised image buffer which is copied from outputImage
	*@return char* 
	*/
	char* getOuptputImage();


	/**
	*prework testing function with Non local mean algorithm
	*/
	void run(){
		test.NonLocalMean(context, command_queue, &this->inputImage,&this->outputImage,this->device,this->imageWidth,this->imageHeight,this->denoisedImageBuffer);
		this->hasImageBuffer = true;
	}

	void test_func();

	void test_func2();
};

