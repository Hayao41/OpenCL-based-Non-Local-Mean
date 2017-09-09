#include "getDevice.h"

#define PLAT_ID 1

void CreateContext(cl_context* context, cl_device_id * device, cl_command_queue* command_queue){
	cl_platform_id *platform;
	cl_uint num_platform;
	cl_int err;

	err = clGetPlatformIDs(0, NULL, &num_platform);
	platform = (cl_platform_id *)malloc(sizeof(cl_platform_id)* num_platform);

	err = clGetPlatformIDs(num_platform, platform, NULL);
	for (int i = 0; i < num_platform; i++)
	{
		size_t size;

		// get name
		err = clGetPlatformInfo(platform[i], CL_PLATFORM_NAME, 0, NULL, &size);
		char *name = (char *)malloc(size);
		err = clGetPlatformInfo(platform[i], CL_PLATFORM_NAME, size, name, NULL);
		printf("CL_PLATFORM_NAME:%s\n", name);

		// vendor
		err = clGetPlatformInfo(platform[i], CL_PLATFORM_VENDOR, 0, NULL, &size);
		char *vendor = (char *)malloc(size);
		err = clGetPlatformInfo(platform[i], CL_PLATFORM_VENDOR, size, vendor, NULL);
		printf("CL_PLATFORM_VENDOR:%s\n", vendor);

		// version
		err = clGetPlatformInfo(platform[i], CL_PLATFORM_VERSION, 0, NULL, &size);
		char *version = (char *)malloc(size);
		err = clGetPlatformInfo(platform[i], CL_PLATFORM_VERSION, size, version, NULL);
		printf("CL_PLATFORM_VERSION:%s\n", version);

		// profile
		err = clGetPlatformInfo(platform[i], CL_PLATFORM_PROFILE, 0, NULL, &size);
		char *profile = (char *)malloc(size);
		err = clGetPlatformInfo(platform[i], CL_PLATFORM_PROFILE, size, profile, NULL);
		printf("CL_PLATFORM_PROFILE:%s\n", profile);

		// extensions
		err = clGetPlatformInfo(platform[i], CL_PLATFORM_EXTENSIONS, 0, NULL, &size);
		char *extensions = (char *)malloc(size);
		err = clGetPlatformInfo(platform[i], CL_PLATFORM_EXTENSIONS, size, extensions, NULL);
		printf("CL_PLATFORM_EXTENSIONS:%s\n", extensions);

		//
		printf("\n\n");

		// clean
		free(name);
		free(vendor);
		free(version);
		free(profile);
		free(extensions);

	}

	cl_uint num_platforms = 0;
	cl_int clStatus = clGetPlatformIDs(0, NULL, &num_platforms);
	cl_device_id* device_list = NULL;
	cl_uint num_device = 0;

	clStatus = clGetDeviceIDs(platform[PLAT_ID], CL_DEVICE_TYPE_GPU, 0, NULL, &num_device);
	device_list = (cl_device_id*)malloc(sizeof(cl_device_id)*num_device);
	clStatus = clGetDeviceIDs(platform[PLAT_ID], CL_DEVICE_TYPE_GPU, num_device, device_list, NULL);

	*context = clCreateContext(NULL, num_device, device_list, NULL, NULL, &clStatus);

	*command_queue = clCreateCommandQueue(*context, device_list[0], 0, &clStatus);
	*device = device_list[0];
}

cl_context CreateContext()
{
    cl_int errNum;
    cl_uint numPlatforms;
    cl_platform_id firstPlatformId;
    cl_context context = NULL;

    // 选择OpenCL平台
    errNum = clGetPlatformIDs(1, &firstPlatformId, &numPlatforms);
    if (errNum != CL_SUCCESS || numPlatforms <= 0)
    {
        std::cerr << "Failed to find any OpenCL platforms." << std::endl;
        return NULL;
    }

    // 在OpenCL平台上创建一个队列，先试GPU，再试CPU
    cl_context_properties contextProperties[] =
    {
        CL_CONTEXT_PLATFORM,
        (cl_context_properties)firstPlatformId,
        0
    };
    context = clCreateContextFromType(contextProperties, CL_DEVICE_TYPE_GPU,
        NULL, NULL, &errNum);
    if (errNum != CL_SUCCESS)
    {
        std::cout << "Could not create GPU context, trying CPU..." << std::endl;
        context = clCreateContextFromType(contextProperties, CL_DEVICE_TYPE_CPU,
            NULL, NULL, &errNum);
        if (errNum != CL_SUCCESS)
        {
            std::cerr << "Failed to create an OpenCL GPU or CPU context." << std::endl;
            return NULL;
        }
    }

    return context;
}

cl_command_queue CreateCommandQueue(cl_context context, cl_device_id *device)
{
    cl_int errNum;
    cl_device_id *devices;
    cl_command_queue commandQueue = NULL;
    size_t deviceBufferSize = -1;

    // First get the size of the devices buffer  
    errNum = clGetContextInfo(context, CL_CONTEXT_DEVICES, 0, NULL, &deviceBufferSize);
    if (errNum != CL_SUCCESS)
    {
        std::cerr << "Failed call to clGetContextInfo(...,GL_CONTEXT_DEVICES,...)";
        return NULL;
    }

    if (deviceBufferSize <= 0)
    {
        std::cerr << "No devices available.";
        return NULL;
    }

    // Allocate memory for the devices buffer  
    devices = new cl_device_id[deviceBufferSize / sizeof(cl_device_id)];
    errNum = clGetContextInfo(context, CL_CONTEXT_DEVICES, deviceBufferSize, devices, NULL);
    if (errNum != CL_SUCCESS)
    {
        std::cerr << "Failed to get device IDs";
        return NULL;
    }

    // In this example, we just choose the first available device.  In a  
    // real program, you would likely use all available devices or choose  
    // the highest performance device based on OpenCL device queries  
    commandQueue = clCreateCommandQueue(context, devices[0], 0, NULL);
    if (commandQueue == NULL)
    {
        std::cerr << "Failed to create commandQueue for device 0";
        return NULL;
    }

    *device = devices[0];
    delete[] devices;
    return commandQueue;
}

//  Create an OpenCL program from the kernel source file  
//  
cl_program CreateProgram(cl_context context, cl_device_id device, std::string fileName)
{
    cl_int errNum;
    cl_program program;

    std::ifstream kernelFile(fileName, std::ios::in);
    if (!kernelFile.is_open())
    {
        std::cerr << "Failed to open file for reading: " << fileName << std::endl;
        return NULL;
    }

    std::ostringstream oss;
    oss << kernelFile.rdbuf();

    std::string srcStdStr = oss.str();
    const char *srcStr = srcStdStr.c_str();
    program = clCreateProgramWithSource(context, 1,
        (const char**)&srcStr,
        NULL, NULL);
    if (program == NULL)
    {
        std::cerr << "Failed to create CL program from source." << std::endl;
        return NULL;
    }

    errNum = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if (errNum != CL_SUCCESS)
    {
        // Determine the reason for the error  
        char buildLog[16384];
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG,
            sizeof(buildLog), buildLog, NULL);

        std::cerr << "Error in kernel: " << std::endl;
        std::cerr << buildLog;
        clReleaseProgram(program);
        return NULL;
    }

    return program;
}

cl_bool ImageSupport(cl_device_id device)
{
    cl_bool imageSupport = CL_FALSE;
    clGetDeviceInfo(device, CL_DEVICE_IMAGE_SUPPORT, sizeof(cl_bool),
        &imageSupport, NULL);
    return imageSupport;
}