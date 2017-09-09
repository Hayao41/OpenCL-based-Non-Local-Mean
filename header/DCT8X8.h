#ifdef _APPLE_
#include <OpenCL/cl.h>
#else
#include <CL\cl.h>
#include <CL\cl.hpp>
#endif

#ifndef __DCT8X8_H__
#define __DCT8X8_H__
class DCT8X8{
private:
public:
	DCT8X8();
	~DCT8X8();
};

#endif