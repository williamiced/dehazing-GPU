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
			("input,i", value<string>()->notifier(checkInputPathValid), "Input image");

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

	return 0;
}
