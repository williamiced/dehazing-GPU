#include "softmatting.h"

float* 	gImgScaleGPU;
float* 	gRefineGPU;
float* 	gN;
float* 	gMeanI;
float* 	gCovI;
float* 	gInvCovI;
int*	gCsrRowPtr;
int* 	gCsrColInd;
float* 	gLapVal;
int 	gRowEleCount;
int 	gNNZ;

__global__ void scaleImg(float* ori, float* out, int width, int channels, int height) {
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	const int i = y * width + x;
	if(x < width && y < height) {
		for (int c=0; c<channels; c++)
			out[i * channels + c] = ori[i * channels + c] / 255.f;
	}
}

__global__ void getNMatrix(float* N, int width, int height, int window) {
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	const int i = y * width + x;
	if(x < width && y < height) {
		float total = 0.f;
		int half = window-1;
		for(int cx = x - half/2; cx <= x + half/2; cx++) {
			for(int cy = y - half/2; cy <= y + half/2; cy++) {
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
		int half = window-1;
		for(int cx = x - half/2; cx <= x + half/2; cx++) {
			for(int cy = y - half/2; cy <= y + half/2; cy++) {
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

		int half = window-1;
		for(int cx = x - half/2; cx <= x + half/2; cx++) {
			for(int cy = y - half/2; cy <= y + half/2; cy++) {
				if(IN_GRAPH(cx, cy, width, height)) {
					int newI = cy * width + cx;
					for (int c1=0; c1<channels; c1++) {
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

__global__ void calcLaplacian(float* I, float* N, float* Mu, float* InvTerm, int* csrRowPtr, int* csrColInd, float* L, int width, int height, int channels, int window) {
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	const int j = blockIdx.y * blockDim.y + threadIdx.y;

	const int ix = i % width;
	const int iy = i / width;
	const int jx = j % width;
	const int jy = j / width;

	const int size = width * height;

	// I must be smaller than J to get upper triangular Laplacian matrix
	// As a result, we know jy <= iy
	if (i >= size || j >= size || i < j) 
		return;

	// If distance of I and J is too large, that means there will be no window cover both I and J, so could be ignored
	if (abs(ix-jx) >= window || abs(iy-jy) >= window) 
		return;

	float total = 0.f;
	float* Ii = I + i * channels;
	float* Ij = I + j * channels;
	int leftX = ix < jx ? ix : jx;
	int rightX = (leftX == ix ? jx : ix);

	int channelsSquare = channels * channels;
	int half = (window-1)/2;

	int count = 0;
	for (int x = max(rightX - 2 * half, 0); x <= min(leftX + 2 * half, width-1); x++) {
		for (int y = max(iy - 2 * half, 0); y <= min(jy + 2 * half, height-1); y++) {
			count++;
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
					tmpVal += (Ii[c2] - Mu[k*channels + c2]) * InvTerm[k*channelsSquare + c2*channels + c1];
				currentTerm += tmpVal * (Ij[c1] - Mu[k*channels + c1]);
			}
			total += delta - invWeight * ( 1 + currentTerm );
		}
	}

	// Get location of L
	bool isFind = false;
	for (int idx=csrRowPtr[i]; idx<csrRowPtr[i+1]; idx++) {
		if (csrColInd[idx] == j) {
			if (i == j)
				L[idx] = PARAM_LAMBDA + total;
			else
				L[idx] = total;
			isFind = true;
			break;
		} 
	}
	if (!isFind) 
		printf("WTF: find %d from ColInd: %d ~ %d\n", j, csrRowPtr[i], csrRowPtr[i+1]);
}

void softMattingMemInit() {
	gRefineGPU	= g1ChannelContainerGPU[2];
	gN 			= g1ChannelContainerGPU[3];

	CUDA_CHECK_RETURN( cudaMalloc((void**)&gImgScaleGPU, gImgWidth * gImgHeight * gImgChannels * sizeof(float) ) );

	// 3 * 1
	CUDA_CHECK_RETURN( cudaMalloc((void**)&gMeanI, gImgWidth * gImgHeight * gImgChannels * sizeof(float) ) );

	// 3 * 3
	CUDA_CHECK_RETURN( cudaMalloc((void**)&gCovI, gImgWidth * gImgHeight * gImgChannels * gImgChannels * sizeof(float) ) );
	CUDA_CHECK_RETURN( cudaMalloc((void**)&gInvCovI, gImgWidth * gImgHeight * gImgChannels * gImgChannels * sizeof(float) ) );

	int maxNNZ = gImgWidth * gImgHeight * (WINDOW_SM * 4 + 1) * (WINDOW_SM * 4 + 1);	
	// Sparse Matrix Structure
	CUDA_CHECK_RETURN( cudaMalloc((void**)&gCsrRowPtr, (gImgWidth * gImgHeight + 1) * sizeof(int) ) );	
	CUDA_CHECK_RETURN( cudaMalloc((void**)&gCsrColInd, maxNNZ * sizeof(int) ) );
	CUDA_CHECK_RETURN( cudaMalloc((void**)&gLapVal, maxNNZ * sizeof(float) ) );
}

void softMattingMemDestroy() {
	CUDA_CHECK_RETURN( cudaFree(gImgScaleGPU) );
	CUDA_CHECK_RETURN( cudaFree(gMeanI) );
	CUDA_CHECK_RETURN( cudaFree(gCovI) );
	CUDA_CHECK_RETURN( cudaFree(gInvCovI) );
	CUDA_CHECK_RETURN( cudaFree(gCsrRowPtr) );
	CUDA_CHECK_RETURN( cudaFree(gCsrColInd) );
	CUDA_CHECK_RETURN( cudaFree(gLapVal) );
}

void preCalcForSoftMatting() {
	int m = gImgWidth * gImgHeight;
	int half = (WINDOW_SM-1) / 2;
	
	// Calculate RowPtr and ColInd
	std::vector<int> rowPtr;
	std::vector<int> colInd;

	gNNZ = 0;
	rowPtr.push_back(gNNZ);
	for (int i=0; i<m; i++) {
		int ix = i % gImgWidth;
		int iy = i / gImgWidth;
		int minx = max(ix - 2 * half, 0);
		int maxx = min(ix + 2 * half, gImgWidth-1);
		int miny = max(iy - 2 * half, 0);
		int maxy = min(iy + 2 * half, gImgHeight-1);
		
		bool needStop = false;
		for (int y=miny; y<=maxy; y++) {
			for (int x=minx; x<=maxx; x++) {
				int cur = y * gImgWidth + x;
				colInd.push_back(cur);
				gNNZ++;
				if (cur == i) {
					needStop = true;
					break;
				}
			}
			if (needStop)
				break;
		}
		rowPtr.push_back(gNNZ);
	}

	CUDA_CHECK_RETURN( cudaMemcpy(gCsrRowPtr, &rowPtr[0], sizeof(int) * (m+1), cudaMemcpyHostToDevice) );
	CUDA_CHECK_RETURN( cudaMemcpy(gCsrColInd, &colInd[0], sizeof(int) * gNNZ, cudaMemcpyHostToDevice) );

	printf("gNNZ: %d\n", gNNZ);
}

void show2Darray(float* arrGPU, int size, int width, int height, int channels) {
	int total = width * height * channels;
	float* arrCPU = (float*)	malloc(sizeof(float) * total);
	cudaMemcpy(arrCPU, arrGPU, sizeof(float) * total, cudaMemcpyDeviceToHost);
	for (int y=0; y<size; y++) {
		for (int c=0; c<channels; c++) {
				printf("%d:%d\t", y, c);
			for (int x=0; x<size; x++) {
				printf("%f\t", arrCPU[channels * (y * width + x) + c]);
			}
			printf("\n");
		}
		
	}
	printf("\n");
	free(arrCPU);
}

void showLaplacian() {
	float* 	cpuLap = (float*)	malloc(sizeof(float) * gNNZ);
	int* 	cpuRow = (int*)		malloc(sizeof(int) * gImgWidth * gImgHeight);
	int*	cpuCol = (int*)		malloc(sizeof(int) * gNNZ);
	cudaMemcpy(cpuLap, gLapVal, sizeof(float) * gNNZ, cudaMemcpyDeviceToHost);
	cudaMemcpy(cpuRow, gCsrRowPtr, sizeof(int) * gImgWidth * gImgHeight, cudaMemcpyDeviceToHost);
	cudaMemcpy(cpuCol, gCsrColInd, sizeof(int) * gNNZ, cudaMemcpyDeviceToHost);

	printf("Laplacian: \n");
	for (int y=0; y<10; y++) {
		int offset = 0;
		for (int x=0; x<10; x++) {
			if (cpuCol[cpuRow[y] + offset] == x) {
				printf("%f\t", cpuLap[cpuRow[y] + offset]);
				offset++;
			} else {
				printf("%f\t", 0.f);
			}
		}
		printf("\n");
	}
}

void solveLinearSystem() {
	int m = gImgWidth * gImgHeight;
	float alpha = (float) PARAM_LAMBDA;

	cusparseHandle_t cuSparseHandle;
	cusparseMatDescr_t descr;
	cusparseSolveAnalysisInfo_t info;

	assert( CUSPARSE_STATUS_SUCCESS == cusparseCreate (&cuSparseHandle) );
	assert( CUSPARSE_STATUS_SUCCESS == cusparseCreateMatDescr (&descr) );
	assert( CUSPARSE_STATUS_SUCCESS == cusparseSetMatFillMode (descr, CUSPARSE_FILL_MODE_LOWER) );
	assert( CUSPARSE_STATUS_SUCCESS == cusparseSetMatIndexBase (descr, CUSPARSE_INDEX_BASE_ZERO) );
	assert( CUSPARSE_STATUS_SUCCESS == cusparseSetMatType (descr, CUSPARSE_MATRIX_TYPE_TRIANGULAR) );
	assert( CUSPARSE_STATUS_SUCCESS == cusparseCreateSolveAnalysisInfo (&info) );

	cudaMemset(gRefineGPU, 0.f, sizeof(float) * m);

	cusparseStatus_t status;
	status = cusparseScsrsv_analysis(cuSparseHandle, 
		CUSPARSE_OPERATION_NON_TRANSPOSE,
		m, // m
		gNNZ, // nnz
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
	CHECK

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
	CHECK

	if (status != CUSPARSE_STATUS_SUCCESS) {
		printf("Solve error %d\n", status);
		exit(-1);
	}

	showLaplacian();
}

void refineTransmission() {
	SETUP_TIMER

	dim3 bdim(BLOCK_DIM, BLOCK_DIM);
	int grid_size_x = CEIL(double(gImgWidth) / BLOCK_DIM);
	int grid_size_y = CEIL(double(gImgHeight) / BLOCK_DIM);
	dim3 gdim(grid_size_x, grid_size_y);

	scaleImg<<<gdim, bdim>>>(gImgGPU, gImgScaleGPU, gImgWidth, gImgHeight, gImgChannels);
	CHECK

	printf("scaleImg: \n");
	show2Darray(gImgScaleGPU, 5, gImgWidth, gImgHeight, gImgChannels);

	getNMatrix<<<gdim, bdim>>>(gN, gImgWidth, gImgHeight, WINDOW_SM);
	CHECK

	printf("N: \n");
	show2Darray(gN, 3, gImgWidth, gImgHeight, 1);

	getMeanMatrix<<<gdim, bdim>>>(gImgScaleGPU, gMeanI, gN, gImgWidth, gImgHeight, gImgChannels, WINDOW_SM);
	CHECK

	printf("MeanI: \n");
	show2Darray(gMeanI, 3, gImgWidth, gImgHeight, 3);

	getCovMatrix<<<gdim, bdim>>>(gImgScaleGPU, gMeanI, gCovI, gN, gImgWidth, gImgHeight, gImgChannels, WINDOW_SM);
	CHECK

	printf("CovI: \n");
	show2Darray(gCovI, 3, gImgWidth, gImgHeight, 9);

	calcInvCovTerm<<<gdim, bdim>>>(gCovI, gInvCovI, gN, gImgWidth, gImgHeight, gImgChannels, WINDOW_SM);
	CHECK

	printf("InvCovI: \n");
	show2Darray(gInvCovI, 3, gImgWidth, gImgHeight, 9);

	CUDA_CHECK_RETURN( cudaMemset(gLapVal, 0.f, sizeof(float) * gNNZ) );

	dim3 bdim_Lap(BLOCK_DIM, BLOCK_DIM);
	int gridSize = CEIL(double(gImgWidth * gImgHeight) / BLOCK_DIM);
	dim3 gdim_Lap(gridSize, gridSize);
	calcLaplacian<<<gdim_Lap, bdim_Lap>>>(gImgScaleGPU, gN, gMeanI, gInvCovI, gCsrRowPtr, gCsrColInd, gLapVal, gImgWidth, gImgHeight, gImgChannels, WINDOW_SM);
	CHECK

	solveLinearSystem();
}


void fillRefineData(float* cpuData) {
	CUDA_CHECK_RETURN( cudaMemcpy(cpuData, gRefineGPU, gImgWidth * gImgHeight * sizeof(float), cudaMemcpyDeviceToHost) );
}	