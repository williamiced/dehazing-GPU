#ifndef MACRO_H_
#define MACRO_H_

#define IN_GRAPH(x,y,w,h) ((x>=0)&&(x<w)&&(y>=0)&&(y<h))
#define min(x,y) ((x<y)?x:y)
#define max(x,y) ((x>y)?x:y)
#define WINDOW 7
#define WINDOW_GF 15
#define WINDOW_SM 3
#define PARAM_OMEGA 0.95
#define PARAM_T0 0.1
#define PARAM_EPSILON 0.000001
#define PARAM_LAMBDA 0.0001

#define CONTAINER_AMOUNT 10

#define __DEBUG

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

#endif // MACRO_H__