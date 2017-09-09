#include "getDevice.h"
#include "LoadImage.h"
#include "Utils.h"
#include "BM3D.h"
#include <time.h>
#include <opencv2/core/core.hpp>        // Basic OpenCV structures (cv::Mat, Scalar)
#include <opencv2/highgui/highgui.hpp>  // OpenCV window I/O
using namespace std;

int main(int argc, char** argv){

	BM3DFilter filter = BM3DFilter();
	cl_context context = 0;
	cl_command_queue commandQueue = 0;
	cl_device_id device = 0;

	CreateContext(&context, &device, &commandQueue);

	// 确保计算设备能够支持图片  
	if (ImageSupport(device) != CL_TRUE)
	{
		cerr << "OpenCL device does not support images." << endl;
		//Cleanup(context, commandQueue, program, kernel, imageObjects, sampler);
		cin.get();
		return 1;
	}

	//初始化滤波器，将平台信息装载入滤波器
	filter.setDevice(device);
	filter.setContext(context);
	filter.setCommandQueue(commandQueue);

	// 将图片载入OpenCL设备
	//string src0 = "images/portrait_noise.bmp";
	//string src0 = "images/Port_pca_big.png";
	//string src0 = "images/noise.bmp";
	//string src0 = "additive_noise.jpg";
	//string src0 = "noise.jpg";
	//string src0 = "cat_noise - 100032538 - orig.jpg";
	//string src0 = "images/flower.jpg";
	//string src0 = "images/kuma.jpg";
	//string src0 = "images/woman.jpg";
	//string src0 = "images/sunset.jpg";
	//string src0 = "images/doge.png";
	//string src1 = "images/dog.jpeg";
	//string src0 = "images/jiangan.jpg";
	string src1 = "images/pic/cat.jpg";
	string src0 = "images/pic/cat.jpg";
	//string src0 = "images/ghibli.jpg";
	filter.setUrl(src0);
	filter.run();
	//filter.test_func();
	//filter.test_func2();

	cv::Mat imageColor = cv::imread(src0);
	cv::imshow("OpenCV Shows Window", imageColor);
	char* buffer = filter.getOuptputImage();
	if (buffer != NULL){
		cv::Mat imageColor1 = cv::imread(src0);
		cv::Mat imageColor2;
		imageColor2.create(imageColor.rows, imageColor.cols, imageColor1.type());
		int w = 0;
		for (int v = imageColor2.rows - 1; v >= 0; v--)
		{
			for (int u = 0; u <imageColor2.cols; u++)
			{
				imageColor2.at<cv::Vec3b>(v, u)[0] = buffer[w++];
				imageColor2.at<cv::Vec3b>(v, u)[1] = buffer[w++];
				imageColor2.at<cv::Vec3b>(v, u)[2] = buffer[w++];
				w++;
			}
		}
		cv::imshow("OpenCL Denosied Window", imageColor2);
		cout << "PSNR:      " << getPSNR(cv::imread(src1), imageColor2) << endl;
	}
	
	cv::waitKey(0);
	delete[] buffer;
	//Cleanup(context, commandQueue, program, kernel, imageObjects, sampler);
	filter.~BM3DFilter();
	return 0;
}

