#include "header\config.h"
#define MAX_BLOCKS 32
#define DIST_THRESHOLD 10.0f

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
    						__global short* similar_coords,		//similar blocks' coordinate record array
							__global uchar* block_counts		//similar blocks counts for every patch block
							){
	__local float   dists[BLOCK_SIZE * BLOCK_SIZE];
	__local float   distances[MAX_BLOCKS];
	__local short2  positions[MAX_BLOCKS];
	__local int   block_count;

    //block's iamge coordinate(top left pixel's coordinate for example : block(0,0)'s coordinate is local coodinate(0,0) and the block(1,0) is the (8,0))
	const int x = get_local_size(0) * get_group_id(0);
	const int y = get_local_size(1) * get_group_id(1);

	//global coordinate contained by it's block in the image 
	const int ix = x + get_local_id(0);
	const int iy = y + get_local_id(1);

	//patch block's center coordinate
	const int bx = x + PATCH_WINDOW_HALF;
	const int by = y + PATCH_WINDOW_HALF;

	//block index in the image blocks
	/*
	*   * * *  *  *
	*   * * *  *  *
	*   * * *  BT *
	*/
	const int bindex = get_group_id(0) + get_group_id(1) * get_num_groups(0);

	//the index in the threads block
	/*
	*   * * *  *  *
	*   * * *  *  *
	*   * * *  TI *
	*/
	const int tindex = get_local_id(0) + get_local_id(1) * get_local_size(0);



    block_count = 0;
    dists[tindex] = 0.0f;
    
	if(tindex < MAX_BLOCKS){
		distances[tindex] = FLT_MAX;
	}
	barrier(CLK_LOCAL_MEM_FENCE);
	
	for (int j = -SEARCH_WINDOW_HALF; j < SEARCH_WINDOW_HALF - PATCH_WINDOW_HALF; j += SLIPPINF_STEP){
		for (int i = -SEARCH_WINDOW_HALF ; i < SEARCH_WINDOW_HALF - PATCH_WINDOW_HALF; i += SLIPPINF_STEP){

			/*put the refrence block into the record stack*/
			if (j == -PATCH_WINDOW_HALF && i == -PATCH_WINDOW_HALF){
				block_count++;
				positions[0].x = -PATCH_WINDOW_HALF;
				positions[0].y = -PATCH_WINDOW_HALF;
				barrier(CLK_LOCAL_MEM_FENCE);
			}
			else{
				float dist = 0.0f;

				//refrence block coordinate in the searching window(top left)
				int bcx = bx + i;
				int bcy = by + j;

				//pixel coordinate int the refrence block
				int cx = bcx + get_local_id(0);
				int cy = bcy + get_local_id(1);

				//caculate distance between two pixels
				dist = vecLen(
					(float3)read_imagef(src, sampler, (int2)(ix, iy)).xyz,
					(float3)read_imagef(src, sampler, (int2)(cx, cy)).xyz
					);

				/*
				*sum all pixel's distance in the block
				*use prefix sum or sum reduction to get the distance sum
				*/
				dists[tindex] = dist;
				barrier(CLK_LOCAL_MEM_FENCE);

				/*
				*parallel sum reduction : group two elements(every thread which's tid under the stride operates it)in the sum buffer then
				*2X2 addition starts
				*
				*localsum[]
				*0  1  2  3  4  5  ....... 31 | 32 33 ....... 60 61 62 63
				*|  |  |  |  |  |          |
				*32 33 34 35 36 37 ....... 63
				*
				*............
				*
				*0 | 1
				*|
				*1
				*|
				*sum = localsum[0]
				*/
				for (uint stride = BLOCK_SIZE * BLOCK_SIZE / 2; stride > 0; stride /= 2){

					//waitting for each 2X2 addition into buffer 
					barrier(CLK_LOCAL_MEM_FENCE);

					//add elements 2 by 2 between tindex and tindex + stride
					if (tindex < stride)
						dists[tindex] += dists[tindex + stride];
				}
				barrier(CLK_LOCAL_MEM_FENCE);

				if (dists[0] <= DIST_THRESHOLD) {
					for (int n = 1; n < MAX_BLOCKS; n++) {
						if (dists[0] <= distances[n]) {
							/* insert the new distance into the distance stack */

							// parallel moving the distances[i]-distance[MAX_BLOCKS-2] to distance[i+1]-distance[MAX_BLOCKS-1]
							// then insert the new distance into distances[i](position do as well as distance)
							/*
							*tindex : the index in the threads block
							*
							*Temp stack for moving synchronize
							*
							*Distance
							*0 1 2 3 ... i   i+1 ..... n-2  n-1(tail)
							*Dist_Temp   |    |         |
							*.......... i+1  i+2 ..... n-1(tail)
							*Distance    |    |         |
							*.......... i+1  i+2 ..... n-1(tail)
							*
							* =
							*
							*Distance
							*0 1 2 3 ... i   i+1 ..... n-2  n-1(tail)
							*Distance    |    |         |
							*.......... i+1  i+2 ..... n-1(tail)
							*
							*then
							*insert new distcance
							*Distance[i] = dists[0]
							*
							*/

							float dist_temp = 0.0f;
							short2 pos_temp;
							if (tindex > n && tindex < MAX_BLOCKS){
								dist_temp = distances[tindex - 1];
								pos_temp = positions[tindex - 1];

								barrier(CLK_LOCAL_MEM_FENCE);

								distances[tindex] = dist_temp;
								positions[tindex] = pos_temp;

								barrier(CLK_LOCAL_MEM_FENCE);

								if (tindex == n + 1){
									block_count++;
									distances[n] = dists[0];
									positions[n].x = i;
									positions[n].y = j;
								}
							}

							barrier(CLK_LOCAL_MEM_FENCE);
							break;
						}
					}
				}
			}
			barrier(CLK_LOCAL_MEM_FENCE);
		}
	}

	barrier(CLK_LOCAL_MEM_FENCE);

	//if block_count upper MAX_BLOCKS set it to MAX_BLOCKS
	if(block_count > MAX_BLOCKS){
		block_count = MAX_BLOCKS;
	}
		
	block_counts[bindex] = block_count;

	//record all blocks' similar blocks' coordinates(top left) in the record array
    //coodinates of similar blocks' coodinates in the similar_coords(coordinate by thread)
	//every two elements define the similar block's top left coordinate
	/*
	*   SI(0)                            SI(72)
	*   |                                |
	*   *        * * * * * * ............* * * * * * *  
	*   |        |
	*   p[ti].x  p[ti].y
	*   |                                |
	*   Block(0,0)                       Blcok(1,0)
	*   bi = 0                           bi = 1;
	*   |                          
	*   bi * MAX_BLOCKS * 2 + ti
	*/
	if(tindex >= 0 && tindex < block_count){
		
		const int sindex = bindex * MAX_BLOCKS * 2 + tindex * 2;

		similar_coords[sindex] = positions[tindex].x;
		similar_coords[sindex+1] = positions[tindex].y;
	}
}


__kernel void BlockMatching_Test(
							read_only image2d_t src,			//input image(read only)
    						__global short* similar_coords,		//similar blocks' coordinate record array
							__global uchar* block_counts		//similar blocks counts for every patch block
							){
	//block's iamge coordinate(top left pixel's coordinate for example : block(0,0)'s coordinate is local coodinate(0,0) the block(1,0) is the (8,0))
	const int x = get_local_size(0) * get_group_id(0);
	const int y = get_local_size(1) * get_group_id(1);

	//global coordinate contained by it's block in the image 
	const int ix = x + get_local_id(0);
	const int iy = y + get_local_id(1);

	//patch block's center coordinate
	const int bx = x + PATCH_WINDOW_HALF;
	const int by = y + PATCH_WINDOW_HALF;

	//block index in the image blocks
	/*
	*   * * *  *  *
	*   * * *  *  *
	*   * * *  BT *
	*/
	const int bindex = get_group_id(0) + get_group_id(1) * get_num_groups(0);

	//the index in the threads block
	/*
	*   * * *  *  *
	*   * * *  *  *
	*   * * *  TI *
	*/
	const int tindex = get_local_id(0) + get_local_id(1) * get_local_size(0);

	for(int i = 0 ; i < MAX_BLOCKS ; i++){
		similar_coords[bindex * MAX_BLOCKS * 2 + i * 2] = -4;
		similar_coords[(bindex * MAX_BLOCKS * 2) + i * 2 + 1] = -4;
	}

	block_counts[bindex] = MAX_BLOCKS;
}