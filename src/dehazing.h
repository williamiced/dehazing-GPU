#ifndef DEHAZING_H_
#define DEHAZING_H_

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <boost/current_function.hpp>
#include <boost/timer/timer.hpp>

#include "macro.h"

void coreMemInit(int width, int height, int channels, float* rawData);
void coreMemDestroy();
void calcDarkChannel();
void calcAirLight(float* A, float* rawData);
void calcTransmission(float* A);
void doDehaze(float* A);
void fillDarkChannelData(float* cpuData);
void fillTransmissionData(float* cpuData);
void fillDehazeData(float* cpuData);

extern float* 	gImgGPU;
extern float* 	gGrayGPU;

// Real container
extern float**	g1ChannelContainerGPU;

extern float* 	gDarkPixelGPU;
extern float* 	gDarkPatchGPU;
extern float*	gTransPixelGPU;
extern float*	gTransPatchGPU;

extern unsigned int*	gSelectedIdxGPU;

extern int 	gImgWidth;
extern int 	gImgHeight;
extern int 	gImgChannels;

// From soft matting
extern float* gRefineGPU;

#endif /* DEHAZING_H_ */


