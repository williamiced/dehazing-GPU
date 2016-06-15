#ifndef GUIDED_FILTER_H_
#define GUIDED_FILTER_H_

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cufft.h>
#include <cusolverSp.h>
#include <boost/current_function.hpp>
#include <boost/timer/timer.hpp>

#include "macro.h"

void guidedMemInit();
void guidedMemDestroy();
void doGuidedFilter();
void fillGuidedData(float* cpuData);

// From core
extern float* 	gImgGPU;
extern float* 	gGrayGPU;
extern float*	gTransPatchGPU;

extern float**	g1ChannelContainerGPU;

extern int 	gImgWidth;
extern int 	gImgHeight;
extern int 	gImgChannels;

// For Guided Filter
extern float* 	gN_g;
extern float* 	gMeanI_g;
extern float*   gMeanI_t;
extern float*   gSigmai;
extern float*   gCross;
extern float*   gA;
extern float*   gB;
extern float*   gMeanA;
extern float*   gMeanB;

#endif // GUIDED_FILTER_H