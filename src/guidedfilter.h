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

void initMemForGuidedFilter();
void guidedFilter();
void fillGuidedData(float* cpuData);

// From core
extern float* 	gImgGPU;
extern float* 	gGrayGPU;
extern float*	gTransPatchGPU;
extern int 	gImgWidth;
extern int 	gImgHeight;
extern int 	gImgChannels;

// For Guided Filter
extern float* 	gGuidedGPU;
extern float* 	gN_g;
extern float* 	gMeanI_g;
extern float*   gMeanI_t;
extern float*   sigmai;
extern float*   cross;
extern float*   a;
extern float*   b;
extern float*   meanA;
extern float*   meanB;

#endif // GUIDED_FILTER_H
