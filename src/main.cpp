#ifdef __APPLE__
        #include <sys/uio.h>
#else
        #include <sys/io.h>
#endif

// Standard headers
#include <iostream>
#include <time.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>

// 3-rd party headers
#include <opencv2/opencv.hpp>
#include <boost/program_options.hpp>

// Our headers
#include "HazeRemover.h"

using namespace cv;
using namespace std;
using namespace boost::program_options;

unique_ptr<HazeRemover> gHazeRemover;

// Define Const
clock_t start , finish ;
float lambda=0.0001;	//lambda
double _w=0.95;			//w

void checkInputPathValid(string pathName) {
	struct stat info;

	if( stat( pathName.c_str(), &info ) != 0 || (info.st_mode & S_IFDIR) ) {
		cout << "Path name: " << pathName << " is not valid." << endl;
		exit(-1);
	}

	gHazeRemover->setInputFilePath(pathName);
}

void processArgs(int argc, char** argv) {
	try {
		options_description desc{"Options"};
		desc.add_options() 
			("help,h", "Help screen")
			("input,i", value<string>()->notifier(checkInputPathValid), "Input image")
			("output,o", value<string>()->default_value("output.png"), "Output image");

		variables_map vm;
		store(parse_command_line(argc, argv, desc), vm);
		notify(vm);

		if (vm.count("help"))
			cout << desc << endl;
	} catch (const boost::program_options::error &ex) {
		cerr << ex.what() << endl;
	}
}

//Main Function
int main(int argc, char * argv[]) {
	gHazeRemover = unique_ptr<HazeRemover>( new HazeRemover() );
	processArgs(argc, argv);

	gHazeRemover->dehaze();

	/*	
	float *trans = NULL;
	CUDA_CHECK_RETURN(cudaMalloc((void **)(&trans), size * sizeof(float)));

	float *filter = NULL;
	CUDA_CHECK_RETURN(cudaMalloc((void **)(&filter), size * sizeof(float)));
	/////////////////
	printf("height: %d width: %d\n", height, width);

	finish_clock();

	//define the block size and grid size
	cout << "Calculating Dark Channel Prior ..." << endl;
	start_clock();
	dim3 block(blockdim, blockdim);
	int grid_size_x = CEIL(double(height) / blockdim);
	int grid_size_y = CEIL(double(width) / blockdim);
	dim3 grid(grid_size_x, grid_size_y);
	//dark channel: dark
	dark_channel(gpu_image, img_gray, dark, height, width, block, grid);
	finish_clock();

	cout << "Calculating Airlight ..." << endl;
	start_clock();
	dim3 block_air(1024);
	dim3 grid_air(CEIL(double(size) / block_air.x));
	//airlight: gpu_image[height*width]
	air_light(gpu_image, dark, height, width, block_air, grid_air);
	finish_clock();
    
	cout << "Calculating transmission ..." << endl;
	start_clock();
	//t: transmission
	transmission(gpu_image, trans, height, width, block, grid);
	finish_clock();

	cout << "Refining transmission ..." << endl;
	dim3 block_guide(blockdim, blockdim);
	int grid_size_x_guide = CEIL(double(height) / blockdim);
	int grid_size_y_guide = CEIL(double(width) / blockdim);
	dim3 grid_guide(grid_size_x_guide, grid_size_y_guide);
	//filter: guided imaging filter result
	gfilter(filter, img_gray, trans, height, width, block_guide, grid_guide);
	finish_clock();
    
	cout << "Calculating dehaze ..." << endl;
	start_clock();
	dehaze(gpu_image, dark, filter, height, width, block, grid);//dehaze image: ori_image
	finish_clock();
    
	cout << "Copy back to host memory ..." << endl;
	start_clock();
	
	CUDA_CHECK_RETURN(cudaFree(dark));
	
	CUDA_CHECK_RETURN(cudaMemcpy(trans_image, filter, size * sizeof(float), cudaMemcpyDeviceToHost));
	CUDA_CHECK_RETURN(cudaFree(filter));
	
	CUDA_CHECK_RETURN(cudaMemcpy(dark_image, trans, size * sizeof(float), cudaMemcpyDeviceToHost));
	CUDA_CHECK_RETURN(cudaFree(trans));
	
	CUDA_CHECK_RETURN(cudaMemcpy(cpu_image, gpu_image, ((size+1) * 3) * sizeof(float), cudaMemcpyDeviceToHost));
	CUDA_CHECK_RETURN(cudaFree(gpu_image));

	for(int i=0;i<size;i++) {
		trans_image[i] *= 255.f;
		dark_image[i] *= 255.f;
	}

	Mat dest(height, width, CV_32FC3, cpu_image);
	Mat trans_dest(height, width, CV_32FC1, trans_image);
	Mat dark_dest(height, width, CV_32FC1, dark_image);
	
	imwrite(out_name, dest);
	imwrite(trans_name, trans_dest);
	imwrite(dark_name, dark_dest);
	
	free(cpu_image);
	free(trans_image);
	free(dark_image);

	free(cpu_image);
	free(trans_image);
	free(dark_image);
	
	finish_clock();
	*/
	return 0;
}
