#ifndef DEHAZING_H_
#define DEHAZING_H_

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <boost/current_function.hpp>
#include <boost/timer/timer.hpp>

#include "macro.h"

void gpuMemInit(int width, int height, int channels, float* rawData);
void gpuMemDestroy();
void calcDarkChannel();
void calcAirLight(float* A, float* rawData);
void calcTransmission(float* A);
void doDehaze(float* A);
void fillDarkChannelData(float* cpuData);
void fillTransmissionData(float* cpuData);
void fillDehazeData(float* cpuData);

extern float* 	gImgGPU;
extern float* 	gDarkPixelGPU;
extern float* 	gDarkPatchGPU;
extern float* 	gGrayGPU;
extern float*	gTransPixelGPU;
extern float*	gTransPatchGPU;
extern int 	gImgWidth;
extern int 	gImgHeight;
extern int 	gImgChannels;

// From soft matting
//extern float* gGuidedGPU;
extern float* gRefineGPU;

#endif /* DEHAZING_H_ */


