#ifndef SOFT_MATTING_H_
#define SOFT_MATTING_H_

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cufft.h>
#include <cusolverSp.h>
#include <boost/current_function.hpp>
#include <boost/timer/timer.hpp>

#include "macro.h"

void refineTransmission();
void refineTransmissionLaplacian();
void preCalcForSoftMatting();
void initMemForSoftMatting();
void fillRefineData(float* cpuData);

// From core
extern float* 	gImgGPU;
extern float* 	gGrayGPU;
extern float*	gTransPatchGPU;
extern int 	gImgWidth;
extern int 	gImgHeight;
extern int 	gImgChannels;

// For soft matting
extern float* gRefineGPU;
extern float* gN;
extern float* gMeanI;
extern float* gCovI;
extern float* gInvCovI;
extern int*	gCsrRowPtr;
extern int* gCsrColInd;
extern float* gLapVal;

#endif // SOFT_MATTING_H
