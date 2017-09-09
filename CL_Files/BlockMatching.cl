
#include "header\config.h"

inline float vecLen(float3 a, float3 b)
{
	return (
		(b.x - a.x) * (b.x - a.x) +
		(b.y - a.y) * (b.y - a.y) +
		(b.z - a.z) * (b.z - a.z)
		);
}


__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;


/**
*Block matching
*
*/
__kernel void BlockMatching(
							read_only image2d_t src,			//input image(read only)
							__global  int* imageW,				//width of image 
							__global  int* imageH,				//height of input image
							__global  float* output
							){
	const int ix = get_local_size(0) * get_group_id(0) + get_local_id(0);
	const int iy = get_local_size(1) * get_group_id(1) + get_local_id(1);

	const int x = ix * STEP_SIZE + PATCH_WINDOW_HALF;
	const int y = iy * STEP_SIZE + PATCH_WINDOW_HALF;

	int block_count = 0;

	float Dist_Stack[MAX_BLOCKS];
	
	for(int n = 0; n < MAX_BLOCKS ; n++){
		Dist_Stack[n] = FLT_MAX;
	}
	float dist = 0.0f;
	int count = 0;
	int index = iy * get_global_size(0) + ix;
	
	for (int j = -SERCH_WINDOW_HALF + PATCH_WINDOW_HALF; j < SERCH_WINDOW_HALF - PATCH_WINDOW_HALF; j += SLIPPINF_STEP){
		for (int i = -SERCH_WINDOW_HALF + PATCH_WINDOW_HALF; i < SERCH_WINDOW_HALF - PATCH_WINDOW_HALF; i += SLIPPINF_STEP){

           dist = 0.0f;
           int column = 0;
           int rows = 0;

		   for(int m = -PATCH_WINDOW_HALF; m < PATCH_WINDOW_HALF ; m++){
				for(int n = -PATCH_WINDOW_HALF; n < PATCH_WINDOW_HALF ; n++){
					float3 ref = (float3)read_imagef(src, sampler, (int2)(x + n,y + m)).xyz;
					float3 fetch = (float3)read_imagef(src, sampler, (int2)(x + i + n,y + j + m)).xyz;
					dist += vecLen(ref,fetch);
				}
				column ++;
			}

			if (1000 <index < 2000 && count < 1 && j == 1 && i == 1){
				output[index] = dist;
				count++;
			}
			/*if(dist < params->DISTANCE_THRESHOLD){

			}*/
		}
	}
	
}