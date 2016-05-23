#CUDA implementation of dehazing algorithm using dark channel prior

Dehazing algorithm implemented on CUDA.

##Feature
- OpenCV to read images and processing them on GPU
- Shared memory optimization
- Multi-platform support (Windows, Linux, Mac)

##Usage

Make sure you have Boost, OpenCV, CUDA toolkit installed and a NVIDIA graphic card
Please modified the environment settings of CUDA if your CUDA version is not 7.5

```sh
git clone https://github.com/williamiced/dehazing-GPU
cd dehazing-GPU
make clean && make
make run
```

**Forked from https://github.com/arsenalliu123/dehazing-GPU**
