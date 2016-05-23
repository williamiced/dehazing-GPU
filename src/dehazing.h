#ifndef DEHAZING_H_
#define DEHAZING_H_

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <vector>
#include <algorithm>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/iterator/counting_iterator.h>
#include <boost/current_function.hpp>
#include <boost/timer/timer.hpp>

#define CUDA_CHECK_RETURN(value) {											\
	cudaError_t _m_cudaStat = value;										\
	if (_m_cudaStat != cudaSuccess) {										\
		fprintf(stderr, "Error %s at line %d in file %s\n",					\
				cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);		\
		exit(1);															\
	} }

#define CHECK {\
	if (cudaSuccess != cudaDeviceSynchronize()) {\
		printf("Failed to do cudaDeviceSynchronize");\
		abort();\
	}\
}

#define SETUP_TIMER std::cout << __FUNCTION__ << std::endl; boost::timer::auto_cpu_timer boostTimer;
#define CEIL(X) ((X-(int)(X)) > 0 ? (int)(X+1) : (int)(X))
#define BLOCK_DIM 32

void dehaze(
	float *image,
	float *dark,
	float *t,
	int height,
	int width,
	dim3 blocks,
	dim3 grids
	);

void gfilter(
	float *filter,
	float *img_gray,
	float *trans,
	int height,
	int width,
	dim3 blocks,
	dim3 grids
	);//filter: guided imaging filter result

void gpuMemInit(int width, int height, int channels, float* rawData);
void gpuMemDestroy();
void calcDarkChannel();
void calcAirLight(float* A, float* rawData);
void calcTransmission(float* A);
void refineTransmission();
void doDehaze(float* A);
void fillDarkChannelData(float* cpuData);
void fillTransmissionData(float* cpuData);
void fillDehazeData(float* cpuData);

extern float* 	gImgGPU;
extern float* 	gDarkPixelGPU;
extern float* 	gDarkPatchGPU;
extern float* 	gGrayGPU;
extern int 	gImgWidth;
extern int 	gImgHeight;
extern int 	gImgChannels;

#endif /* DEHAZING_H_ */


