#include "softmatting.h"

float* 	gRefineGPU;
float* 	gN;
float* 	gMeanI;
float* 	gCovI;
float* 	gInvCovI;
int*	gCsrRowPtr;
int* 	gCsrColInd;
float* 	gLapVal;
int 	gRowEleCount;
cusolverSpHandle_t gSpSolver;

__global__ void getNMatrix(float* N, int width, int height, int window) {
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	const int i = y * width + x;
	if(x < width && y < height) {
		float total = 0.f;
		for(int startx = 0; startx < window * 2 + 1; startx++) {
			for(int starty = 0; starty < window * 2 + 1; starty++) {
				int cx = x-window+startx;
				int cy = y-window+starty;
				if(IN_GRAPH(cx, cy, width, height)) 
					total += 1.f;
			}
		}
		N[i] = total;
	}
}

__global__ void getMeanMatrix(float* I, float* meanI, float* N, int width, int height, int channels, int window) {
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	const int i = y * width + x;
	if(x < width && y < height) {
		float total[3] = {0.f};
		for(int startx = 0; startx < window * 2 + 1; startx++) {
			for(int starty = 0; starty < window * 2 + 1; starty++) {
				int cx = x-window+startx;
				int cy = y-window+starty;
				if(IN_GRAPH(cx, cy, width, height)) {
					int newI = cy * width + cx;
					for (int c=0; c<channels; c++)
						total[c] += I[newI * channels + c];
				}
			}
		}
		for (int c=0; c<channels; c++)
			meanI[i * channels + c] = total[c] / N[i];
	}
}

__global__ void getCovMatrix(float* I, float* meanI, float* covI, float* N, int width, int height, int channels, int window) {
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	const int i = y * width + x;
	if(x < width && y < height) {
		float total[9] = {0.f};
		for(int startx = 0; startx < window * 2 + 1; startx++) {
			for(int starty = 0; starty < window * 2 + 1; starty++) {
				int cx = x-window+startx;
				int cy = y-window+starty;
				if(IN_GRAPH(cx, cy, width, height)) {
					int newI = cy * width + cx;
					for(int c1=0; c1<channels; c1++) {
						for (int c2=c1; c2<channels; c2++) {
							total[c1*channels + c2] += (I[newI * channels + c1] - meanI[i * channels + c1]) 
														* (I[newI * channels + c2] - meanI[i * channels + c2]);
						}
					}
				}
			}
		}
		int channelsSquare = channels * channels;
		for (int c1=0; c1<channels; c1++) {
			for (int c2=0; c2<channels; c2++) {
				if (c1 <= c2)
					covI[i * channelsSquare + c1 * channels + c2] = total[c1 * channels + c2] / N[i];
				else
					covI[i * channelsSquare + c1 * channels + c2] = total[c2 * channels + c1] / N[i];
			}
		}
	}	
}

__global__ void calcInvCovTerm(float* covI, float* invCov, float* N, int width, int height, int channels, int window) {
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	const int i = y * width + x;
	if(x < width && y < height) {
		float* cov = covI + i * channels * channels;
		float* icov = invCov + i * channels * channels;

		// Plus Epsilon U3
		float weight = PARAM_EPSILON / N[i];
		cov[0] += weight;
		cov[4] += weight;
		cov[8] += weight;

		// Find inverse determinant
		float det = cov[0] * (cov[4] * cov[8] - cov[5] * cov[7]) - cov[1] * (cov[3] * cov[8] - cov[5] * cov[6]) + cov[2] * (cov[3] * cov[7] - cov[4] * cov[6]);
		float idet = 1 / det;

		// Calculate inverse factor
		icov[0] = (cov[4] * cov[8] - cov[5] * cov[7]) * idet;
		icov[1] = (cov[2] * cov[7] - cov[1] * cov[8]) * idet;
		icov[2] = (cov[1] * cov[5] - cov[2] * cov[4]) * idet;
		icov[3] = (cov[5] * cov[6] - cov[3] * cov[8]) * idet;
		icov[4] = (cov[0] * cov[8] - cov[2] * cov[6]) * idet;
		icov[5] = (cov[2] * cov[3] - cov[0] * cov[5]) * idet;
		icov[6] = (cov[3] * cov[7] - cov[4] * cov[6]) * idet;
		icov[7] = (cov[1] * cov[6] - cov[0] * cov[7]) * idet;
		icov[8] = (cov[0] * cov[4] - cov[1] * cov[3]) * idet;
	}
}

__global__ void multiplyLambda(float* T, int width, int height) {
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	const int i = y * width + x;
	if(x < width && y < height) {
		T[i] *= PARAM_LAMBDA;
	}
}

__global__ void fillSparseStructure(int* csrRowPtr, int* csrColInd, int width, int height, int channels, int rowEleCount, int w) {
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	const int i = y * width + x;

	w *= 2;
	if (x < width && y < height) {
		// Previous rows total
		int firstRow = rowEleCount * ( min(y, w) * (w+1) + (0 + min( y-1, w-1)) * min(y, w) / 2);
		int mediumRow = rowEleCount * (2*w + 1) * min ( max(y-w, 0), height - 2*w );
		int lastRow = rowEleCount * ( max( w + y - height, 0) * (w+1) + (w-1 + min(height - y, w)) * max( w + y - height, 0) / 2 );

		// Current row position
		int currentRowRange = y < w ? (w+1+y) : (y >= (height-w) ? (w+height-y) : (2*w+1) );

		int firstEle = min(x, w) * (w+1) + (0 + min(x-1, w-1)) * min(x, w) / 2;
		int mediumEle = (2*w+1) * min ( max(x-w, 0), width - 2*w );
		int lastEle = max( w + x - width, 0) * (w+1) + (w-1 + min(width - x, w)) * max( w + x - width, 0) / 2;

		int startIdx = firstRow + mediumRow + lastRow + currentRowRange * (firstEle + mediumEle + lastEle);
		int currentColRange = x < w ? (w+1+x) : (x >= (width-w) ? (w+width-x) : (2*w+1) );

		int lx = max(x-w, 0);
		int ty = max(y-w, 0);

		csrRowPtr[i] = startIdx;

		for(int j = 0; j<currentColRange * currentRowRange; j++) 
			csrColInd[startIdx + j] = (j / currentColRange + ty) * width + (j % currentColRange + lx);
		if (x == width-1 && y == height-1) 
			csrRowPtr[i+1] = startIdx + currentColRange * currentRowRange;
	} 
}

__global__ void calcLaplacian(float* I, float* N, float* Mu, float* InvTerm, int* csrRowPtr, int* csrColInd, float* L, int width, int height, int channels, int window) {
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	const int j = blockIdx.y * blockDim.y + threadIdx.y;

	const int ix = i % width;
	const int iy = i / width;
	const int jx = j % width;
	const int jy = j / width;

	const int size = width * height;
	if (i >= size || j >= size)
		return;

	const int kernelSizeX = 2 * window + 1 - abs(ix-jx);
	const int kernelSizeY = 2 * window + 1 - abs(iy-jy);

	if (kernelSizeX > 0 && kernelSizeY > 0) {
		const float invWeight = 1.f / N[i];

		const int left = min(ix, jx);
		const int right = max(ix, jx);
		const int top = min(iy, jy);
		const int bottom = max(iy, jy);
		float total = 0.f;
		float* Ii = I + i * channels;
		float* Ij = I + j * channels;
		bool isZero = true;

		for(int x=right - window; x <= left + window; x++) {
			for (int y=bottom - window; y <= top + window; y++) {
				isZero = false;
				float delta = (i == j ? 1.f : 0.f);
				float term = 0.f;
				int k = y * width + x;
				for (int c=0; c<channels; c++) {
					float firstStep = 0.f;
					for (int c2=0; c2<channels; c2++)
						firstStep += (Ii[c2] - Mu[k * channels + c2]) * InvTerm[c2 * channels + c];
					term += (Ij[c] - Mu[k * channels + c]) * firstStep;
				}
				total += delta - invWeight * (1 + term);
			}
		}

		if (!isZero) {
			// Get location of L
			// First, I need to know the order of j in the i's window
			int lx = max(ix-2*window, 0);
			int rx = min(ix+2*window, width-1);

			int offset = (rx-lx+1) * (jy-iy) + (jx-ix);
			int idx = csrRowPtr[i] + offset;

			L[idx] = total;

			if (i == j)
				L[idx] += PARAM_LAMBDA;
		} 
	}
}

__global__ void inverseRefinement(float* r, int width, int height) {
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	const int i = y * width + x;
	if(x < width && y < height) {
		r[i] = (255.f - r[i]) / 255.f;
	}
}

void initMemForSoftMatting() {
	// 1 * 1
	CUDA_CHECK_RETURN( cudaMalloc(&gRefineGPU, gImgWidth * gImgHeight * sizeof(float) ) );

	// 1 * 1
	CUDA_CHECK_RETURN( cudaMalloc(&gN, gImgWidth * gImgHeight * sizeof(float) ) );

	// 3 * 1
	CUDA_CHECK_RETURN( cudaMalloc(&gMeanI, gImgWidth * gImgHeight * gImgChannels * sizeof(float) ) );

	// 3 * 3
	CUDA_CHECK_RETURN( cudaMalloc(&gCovI, gImgWidth * gImgHeight * gImgChannels * gImgChannels * sizeof(float) ) );

	// 3 * 3
	CUDA_CHECK_RETURN( cudaMalloc(&gInvCovI, gImgWidth * gImgHeight * gImgChannels * gImgChannels * sizeof(float) ) );

	// nnz * 1 (nnz < size*(4w+1)^2 )
	int nnz = gImgWidth * gImgHeight * (WINDOW_SM * 4 + 1) * (WINDOW_SM * 4 + 1);	

	// Sparse Matrix Structure
	CUDA_CHECK_RETURN( cudaMalloc(&gCsrRowPtr, (gImgWidth * gImgHeight + 1) * sizeof(int) ) );	
	CUDA_CHECK_RETURN( cudaMalloc(&gCsrColInd, nnz * sizeof(int) ) );
	CUDA_CHECK_RETURN( cudaMalloc(&gLapVal, nnz * sizeof(float) ) );
	CUDA_CHECK_RETURN( cudaMemset(gLapVal, 0.f, nnz * sizeof(float)));

	cusolverStatus_t status = cusolverSpCreate(&gSpSolver);
	if (status != CUSOLVER_STATUS_SUCCESS) {
		printf("Solver create error\n");
		exit(-1);
	}
}

void preCalcForSoftMatting() {
	gRowEleCount = 0;
	for (int i=0; i<gImgWidth; i++)
		gRowEleCount += (2*WINDOW_SM+1) + min(i, min(2*WINDOW_SM, gImgWidth-1-i));
}

void refineTransmission() {
	SETUP_TIMER

	dim3 bdim(BLOCK_DIM, BLOCK_DIM);
	int grid_size_x = CEIL(double(gImgWidth) / BLOCK_DIM);
	int grid_size_y = CEIL(double(gImgHeight) / BLOCK_DIM);
	dim3 gdim(grid_size_x, grid_size_y);

	getNMatrix<<<gdim, bdim>>>(gN, gImgWidth, gImgHeight, WINDOW_SM);
	CHECK

	getMeanMatrix<<<gdim, bdim>>>(gImgGPU, gMeanI, gN, gImgWidth, gImgHeight, gImgChannels, WINDOW_SM);
	CHECK

	getCovMatrix<<<gdim, bdim>>>(gImgGPU, gMeanI, gCovI, gN, gImgWidth, gImgHeight, gImgChannels, WINDOW_SM);
	CHECK

	multiplyLambda<<<gdim, bdim>>>(gTransPatchGPU, gImgWidth, gImgHeight);

	if (gImgChannels == 3) {
		calcInvCovTerm<<<gdim, bdim>>>(gCovI, gInvCovI, gN, gImgWidth, gImgHeight, gImgChannels, WINDOW_SM);
		CHECK

		fillSparseStructure<<<gdim, bdim>>>(gCsrRowPtr, gCsrColInd, gImgWidth, gImgHeight, gImgChannels, gRowEleCount, WINDOW_SM);
		CHECK

		dim3 bdim_Lap(BLOCK_DIM, BLOCK_DIM);
		int gridSize = CEIL(double(gImgWidth * gImgHeight) / BLOCK_DIM);
		dim3 gdim_Lap(gridSize, gridSize);
		calcLaplacian<<<gdim_Lap, bdim_Lap>>>(gImgGPU, gN, gMeanI, gInvCovI, gCsrRowPtr, gCsrColInd, gLapVal, gImgWidth, gImgHeight, gImgChannels, WINDOW_SM);
		CHECK

		int nnz;
		cudaMemcpy(&nnz, gCsrRowPtr + gImgWidth * gImgHeight, sizeof(int), cudaMemcpyDeviceToHost);
		
		printf("NNZ: %d\n", nnz);

		cusparseMatDescr_t descr;
		cusparseCreateMatDescr(&descr);
		cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL);
		cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO);
		int singularity = 0;

		cusolverStatus_t status = cusolverSpScsrlsvqr(gSpSolver, 
			gImgWidth * gImgHeight, // n
			nnz, // nnzA
			descr, // cusparseMatDescr_t
			gLapVal,
			gCsrRowPtr,
			gCsrColInd,
			gTransPatchGPU, // b
			1e-5, // tol
			0, // reorder
			gRefineGPU, 
			&singularity);

		inverseRefinement<<<gdim, bdim>>>(gRefineGPU, gImgWidth, gImgHeight);
	}
}


void fillRefineData(float* cpuData) {
	CUDA_CHECK_RETURN( cudaMemcpy(cpuData, gRefineGPU, gImgWidth * gImgHeight * sizeof(float), cudaMemcpyDeviceToHost) );
}	