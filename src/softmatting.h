#ifndef SOFT_MATTING_H_
#define SOFT_MATTING_H_

void refineTransmission();

extern float* 	gGrayGPU;
extern float*	gTransPatchGPU;
extern int 	gImgWidth;
extern int 	gImgHeight;
extern int 	gImgChannels;

#endif // SOFT_MATTING_H
