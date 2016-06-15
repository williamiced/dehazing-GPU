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

	// I must be smaller than J to get upper triangular Laplacian matrix
	// As a result, we know jy >= iy
	if (i >= size || j >= size || i > j) 
		return;

	// If distance of I and J is too large, that means there will be no window cover both I and J, so could be ignored
	if (abs(ix-jx) > 2 * window || abs(iy-jy) > 2 * window) 
		return;

	float total = 0.f;
	float* Ii = I + i * channels;
	float* Ij = I + j * channels;
	int leftX = ix < jx ? ix : jx;
	int rightX = (leftX == ix ? jx : ix);

	for (int x = max(rightX-window, 0); x <= min(leftX+window, width-1); x++) {
		for (int y = max(jy-window, 0); y <= min(iy+window, height-1); y++) {
			float delta = (i == j ? 1.f : 0.f);
			int k = y * width + x; // Current window center
			float invWeight = 1.f / N[k];

			// 								[Inv_0	Inv_1	Inv_2]		[Ij_R]
			// [Ii_R	Ii_G	Ii_B]	x	[Inv_3	Inv_4	Inv_5]	x	[Ij_G]
			//								[Inv_6	Inv_7	Inv_8]		[Ij_B]

			float currentTerm = 0.f;
			for(int c1=0; c1<channels; c1++) {
				float tmpVal = 0.f;
				for (int c2=0; c2<channels; c2++) 
					tmpVal += (Ii[c2] - Mu[k*channels + c2]) * InvTerm[k*channels*channels + c2*channels + c1];
				currentTerm += tmpVal * (Ij[c1] - Mu[k*channels + c1]);
			}
			total += delta - invWeight * ( 1 + currentTerm );
		}
	}

	// Get location of L

	int lx = max(ix-2*window, 0);
	int rx = min(ix+2*window, width-1);
	int ly = max(iy-2*window, 0);

	int offset = (jy-ly) * (rx-lx+1) + (jx-lx); // yOffset * rangeWidth + xOffset
	int idx = csrRowPtr[i] + offset;

	if (i == j)
		L[idx] = PARAM_LAMBDA + total;
	else
		L[idx] = total;
}

__global__ void inverseRefinement(float* r, int width, int height) {
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	const int i = y * width + x;
	if(x < width && y < height) {
		r[i] = (255.f - r[i]) / 255.f;
	}
}

void softMattingMemInit() {
	gRefineGPU	= g1ChannelContainerGPU[2];
	gN 			= g1ChannelContainerGPU[3];

	// 3 * 1
	CUDA_CHECK_RETURN( cudaMalloc((void**)&gMeanI, gImgWidth * gImgHeight * gImgChannels * sizeof(float) ) );

	// 3 * 3
	CUDA_CHECK_RETURN( cudaMalloc((void**)&gCovI, gImgWidth * gImgHeight * gImgChannels * gImgChannels * sizeof(float) ) );
	CUDA_CHECK_RETURN( cudaMalloc((void**)&gInvCovI, gImgWidth * gImgHeight * gImgChannels * gImgChannels * sizeof(float) ) );

	int nnz = gImgWidth * gImgHeight * (WINDOW_SM * 4 + 1) * (WINDOW_SM * 4 + 1);	
	// Sparse Matrix Structure
	CUDA_CHECK_RETURN( cudaMalloc((void**)&gCsrRowPtr, (gImgWidth * gImgHeight + 1) * sizeof(int) ) );	
	CUDA_CHECK_RETURN( cudaMalloc((void**)&gCsrColInd, nnz * sizeof(int) ) );
	CUDA_CHECK_RETURN( cudaMalloc((void**)&gLapVal, nnz * sizeof(float) ) );
}

void softMattingMemDestroy() {
	CUDA_CHECK_RETURN( cudaFree(gMeanI) );
	CUDA_CHECK_RETURN( cudaFree(gCovI) );
	CUDA_CHECK_RETURN( cudaFree(gInvCovI) );
	CUDA_CHECK_RETURN( cudaFree(gCsrRowPtr) );
	CUDA_CHECK_RETURN( cudaFree(gCsrColInd) );
	CUDA_CHECK_RETURN( cudaFree(gLapVal) );
}

void preCalcForSoftMatting() {
	gRowEleCount = 0;
	for (int i=0; i<gImgWidth; i++)
		gRowEleCount += (2*WINDOW_SM+1) + min(i, min(2*WINDOW_SM, gImgWidth-1-i));
}

void refineTransmission() {
	SETUP_TIMER

	int maxNNZ = gImgWidth * gImgHeight * (WINDOW_SM * 4 + 1) * (WINDOW_SM * 4 + 1);	
	CUDA_CHECK_RETURN( cudaMemset(gLapVal, 0.f, maxNNZ * sizeof(float)));
	CUDA_CHECK_RETURN( cudaMemset(gMeanI, 0.f, gImgWidth * gImgHeight * sizeof(float)));
	CUDA_CHECK_RETURN( cudaMemset(gN, 0.f, gImgWidth * gImgHeight * sizeof(float)));

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

		cusparseHandle_t cuSparseHandle;
		cusparseMatDescr_t descr;
		cusparseSolveAnalysisInfo_t info;

		cusparseCreate (&cuSparseHandle);
		cusparseCreateMatDescr (&descr);
		cusparseSetMatIndexBase (descr, CUSPARSE_INDEX_BASE_ZERO);
		cusparseSetMatType (descr, CUSPARSE_MATRIX_TYPE_TRIANGULAR);
		cusparseCreateSolveAnalysisInfo (&info);

		int m = gImgWidth * gImgHeight;

		cusparseStatus_t status;
		status = cusparseScsrsv_analysis(cuSparseHandle, 
			CUSPARSE_OPERATION_NON_TRANSPOSE,
			m, // m
			nnz, // nnz
			descr, // cusparseMatDescr_t
			gLapVal,
			gCsrRowPtr,
			gCsrColInd,
			info
		);

		if (status != CUSPARSE_STATUS_SUCCESS) {
			printf("Analysis error\n");
			exit(-1);
		}

		float alpha = (float) PARAM_LAMBDA;

		status = cusparseScsrsv_solve(cuSparseHandle, 
			CUSPARSE_OPERATION_NON_TRANSPOSE,
			m, // m
			&alpha, // alpha
			descr,
			gLapVal,
			gCsrRowPtr,
			gCsrColInd,
			info, 
			gTransPatchGPU, // b
			gRefineGPU // X
		);

		if (status != CUSPARSE_STATUS_SUCCESS) {
			printf("Solve error %d\n", status);
		}

		float* cpu ;
		cpu = (float*)malloc(sizeof(float) * 20);
		cudaMemcpy(cpu, gRefineGPU, sizeof(float) * 20, cudaMemcpyDeviceToHost);
		for (int i=0; i<20; i++)
			printf("%f\n", cpu[i]);

		inverseRefinement<<<gdim, bdim>>>(gRefineGPU, gImgWidth, gImgHeight);
	}
}


void fillRefineData(float* cpuData) {
	CUDA_CHECK_RETURN( cudaMemcpy(cpuData, gRefineGPU, gImgWidth * gImgHeight * sizeof(float), cudaMemcpyDeviceToHost) );
}	