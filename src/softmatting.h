#ifndef SOFT_MATTING_H_
#define SOFT_MATTING_H_

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <boost/current_function.hpp>
#include <boost/timer/timer.hpp>

#include "macro.h"

void refineTransmission();
void initMemForSoftMatting();
void fillRefineData(float* cpuData);

// From core
extern float* 	gGrayGPU;
extern float*	gTransPatchGPU;
extern int 	gImgWidth;
extern int 	gImgHeight;
extern int 	gImgChannels;

// For soft matting
extern float* gN;
extern float* gA;
extern float* gB;
extern float* gMeanA;
extern float* gMeanB;
extern float* gMeanI;
extern float* gMeanP;
extern float* gII;
extern float* gIP;
extern float* gMeanII;
extern float* gMeanIP;

#endif // SOFT_MATTING_H
