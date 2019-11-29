
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <iomanip>
#include <opencv/cv.hpp>
#include <opencv2/photo/cuda.hpp>
#include "cudaExamp.h"
#include "misc.h"
#include "Vec3.h"
#include <time.h>

#include "Ray.h"
#include "Sphere.h"
#include "Hittable.h"
#include "HittableList.h"
#include "Material.h"
#include "Camera.h"
#include "WorldGen.h"

#include "RTRendererCUDAQT.h"
#include <QtWidgets/QApplication>

constexpr auto ITER = 50;
constexpr auto SPP = 4;

__global__ void cuHelloWorld()
{
	printf("Hello world");
}

extern "C" void launchKernal()
{
	cuHelloWorld <<<1, 1 >>> ();
}

__device__ Vec3 color(const Ray& r, Hittable** world, int depth, curandState* localRandState)
{
	HitRecord rec;
	if ((*world)->hit(r, 0.000001, FLT_MAX, rec, localRandState)) {
		Ray scattered;
		Vec3 attenuation;
		Vec3 emitted = rec.matPtr->emitted(rec.u, rec.v, rec.point);
		if (depth < ITER && rec.matPtr->scatter(r, rec, attenuation, scattered, localRandState)) {
			return emitted + attenuation * color(scattered, world, depth + 1, localRandState);
		}
		else {
			return emitted;
		}
	}
	else {
#ifdef DARKSCENE
		Vec3 c(0, 0, 0);
#else
		Vec3 unit_direction = unitVector(cur_ray.direction);
		double t = 0.5f * (unit_direction.e[1] + 1.0f);
		Vec3 c = (1.0f - t) * Vec3(1.0, 1.0, 1.0) + t * Vec3(0.5, 0.7, 1.0);
#endif
		return c;
	}
}

__device__ Vec3 color(const Ray& r, Hittable** world,curandState* localRandState)
{
	Ray cur_ray = r;
	Vec3 cur_attenuation = Vec3(1.0, 1.0, 1.0);
	for (int i = 0; i < ITER; i++) {
		HitRecord rec;
		if ((*world)->hit(cur_ray, 0.001, FLT_MAX, rec, localRandState)) {
			Ray scattered;
			Vec3 attenuation;
			Vec3 emitted = rec.matPtr->emitted(rec.u, rec.v, rec.point);
			if (rec.matPtr->scatter(cur_ray, rec, attenuation, scattered, localRandState)) {
				cur_attenuation *= attenuation;
				cur_attenuation += emitted;
				cur_ray = scattered;
			}
			else {
				return cur_attenuation * emitted;
			}
		}
		else {
#ifdef DARKSCENE
			Vec3 c(0, 0, 0);
#else
			Vec3 unit_direction = unitVector(cur_ray.direction);
			double t = 0.5f * (unit_direction.e[1] + 1.0f);
			Vec3 c = (1.0f - t) * Vec3(1.0, 1.0, 1.0) + t * Vec3(0.5, 0.7, 1.0);
#endif
			return cur_attenuation * c;
		}
	}
	return Vec3(0.05, 0.05, 0.1); // exceeded recursion
}

// Main rander func.
__global__ void render(int frameCount, double* fBuffer, Camera** cam, Hittable** world, curandState* randState)  //{b, g, r}, stupid opencv
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;

#ifdef ALLOWOUTOFBOUND
	if ((i >= MAX_X) || (j >= MAX_Y)) return;
#endif // OUTOFBOUNDDETECT

	int index = j * MAX_X + i;
	curandState localRandState = randState[index];
	Vec3 pixel(0, 0, 0);
	if (frameCount)
	{
		pixel.readFrameBuffer(i, j, fBuffer);
		pixel = pixel * pixel;
		pixel *= frameCount;
		pixel *= SPP;
	}
	for (int s = 0; s < SPP; s++) {
		double u = double(i + curand_uniform(&localRandState)) / double(MAX_X);
		double v = double(j + curand_uniform(&localRandState)) / double(MAX_Y);
		Ray r = (*cam)->getRay(u, v, &localRandState);
		pixel += color(r, world, &localRandState);
	}
	randState[index] = localRandState;
	pixel /= double(SPP);
	pixel /= frameCount + 1.0;
	pixel.e[0] = sqrt(pixel.e[0]);
	pixel.e[1] = sqrt(pixel.e[1]);
	pixel.e[2] = sqrt(pixel.e[2]);

	pixel.writeFrameBuffer(i, j, fBuffer);
}

__global__ void rander_init(curandState* randState)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	if ((i >= MAX_X) || (j >= MAX_Y)) return;
	int pixel_index = j * MAX_X + i;
	curand_init(2019+pixel_index, 0, 0, &randState[pixel_index]);
}

int main(int argc, char* argv[])
{
	clock_t clk;
	clk = clock();
	double renderTime;

	std::cout << "Rendering a " << MAX_X << "x" << MAX_Y << " image ";
	std::cout << "in " << BLK_X << "x" << BLK_Y << " blocks, SPP = " <<SPP<<" & depth = "<<ITER<<"\n";

	size_t* pValue = new size_t;
	checkCudaErrors(cudaDeviceGetLimit(pValue, cudaLimitStackSize));
	std::cout << "Stack size limit: \t\t\t" << *pValue << "Byte. Resizing to 65536...";

	checkCudaErrors(cudaDeviceSetLimit(cudaLimitStackSize, 1 << 16));
	checkCudaErrors(cudaDeviceGetLimit(pValue, cudaLimitStackSize));
	std::cout << "...Done. \nStack size limit: \t\t\t" << *pValue << "Byte.\n";

	checkCudaErrors(cudaDeviceGetLimit(pValue, cudaLimitPrintfFifoSize));
	std::cout << "printf() fifo limit: \t\t\t" << *pValue << "Byte.\n";
	checkCudaErrors(cudaDeviceGetLimit(pValue, cudaLimitMallocHeapSize));
	std::cout << "Malloc heap size limit: \t\t" << *pValue << "Byte.\n";
	checkCudaErrors(cudaDeviceGetLimit(pValue, cudaLimitDevRuntimeSyncDepth));
	std::cout << "cudaLimitDevRuntimeSyncDepth: \t\t" << *pValue << ".\n";
	checkCudaErrors(cudaDeviceGetLimit(pValue, cudaLimitDevRuntimePendingLaunchCount));
	std::cout << "cudaLimitDevRuntimePendingLaunchCount: \t" << *pValue << ".\n";
	checkCudaErrors(cudaDeviceGetLimit(pValue, cudaLimitMaxL2FetchGranularity));
	std::cout << "cudaLimitMaxL2FetchGranularity: \t" << *pValue << "Byte.\n";

#ifdef _DEBUG
	std::cout << "Warning: Compiled in debug mode and it hurt performance.\n";
#endif

	cv::Mat M(MAX_Y, MAX_X, CV_64FC3, cv::Scalar(0, 0, 0));

	size_t frameBufferSize = 3 * MAX_X * MAX_Y * sizeof(double);
	double* frameBuffer;
	checkCudaErrors(cudaMallocManaged((void**)&frameBuffer, frameBufferSize));

	Hittable** cudaList;
	int num_Hittables = 500;
	checkCudaErrors(cudaMalloc((void**)&cudaList, num_Hittables * sizeof(Hittable*)));
	Hittable** cudaWorld;
	checkCudaErrors(cudaMalloc((void**)&cudaWorld, sizeof(Hittable*)));
	Camera** cudaCam;
	checkCudaErrors(cudaMalloc((void**)&cudaCam, sizeof(Camera*)));
	
	double ms = double(clock() - clk);
	std::cout << "Alloc \t\t@ t+ " << ms << " ms.\r\n";

	curandState* worldGenRandState;
	checkCudaErrors(cudaMalloc((void**)&worldGenRandState, sizeof(curandState)));

	cv::Mat em = cv::imread("earthmap.jpg");
	unsigned char* t;
	checkCudaErrors(cudaMalloc((void**)&t, sizeof(unsigned char) * em.rows * em.cols * 3));
	checkCudaErrors(cudaMemcpy(t, em.data, sizeof(unsigned char) * em.rows * em.cols * 3, cudaMemcpyHostToDevice));

	// createRandScene <<<1, 1 >>> (cudaList, cudaWorld, cudaCam, t, em.cols, em.rows, worldGenRandState);
	// createWorld1 <<<1, 1 >>> (cudaList, cudaWorld, cudaCam, worldGenRandState);
	// createCheckerTest <<<1, 1 >>> (cudaList, cudaWorld, cudaCam, worldGenRandState);
	// createCornellBox << <1, 1 >> > (cudaList, cudaWorld, cudaCam, worldGenRandState);
	createCornellSmoke << <1, 1 >> > (cudaList, cudaWorld, cudaCam, worldGenRandState);

	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	ms = double(clock() - clk);
	std::cout << "WorldGen \t@ t+ " << ms << " ms.\r\n";

	dim3 blocks(MAX_X / BLK_X + 1, MAX_Y / BLK_Y + 1);
	dim3 threads(BLK_X, BLK_Y);

	curandState* renderRandomStates;
	checkCudaErrors(cudaMalloc((void**)&renderRandomStates, MAX_X * MAX_Y * sizeof(curandState)));
	rander_init <<<blocks, threads >>> (renderRandomStates);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	ms = double(clock() - clk);
	double renderStart = ms;
	std::cout << "init rander \t@ t+ " << ms << " ms.\r\n";

	int frameCount = 0;
	while (1)
	{
		renderTime = ms;

		render <<<blocks, threads >>> (frameCount++, frameBuffer, cudaCam, cudaWorld, renderRandomStates);

		checkCudaErrors(cudaGetLastError());
		checkCudaErrors(cudaDeviceSynchronize());

		ms = double(clock() - clk);
		renderTime = ms - renderTime;
		std::cout << std::fixed << std::setprecision(2) << "Render Time: " << renderTime / 1000.0 << " / " << (ms - renderStart) / 1000.0 / frameCount << " / " << (ms - renderStart)/1000.0 << " s, current SPP = " << frameCount * SPP << "\r\n";

		M.data = (uchar*)frameBuffer;
		cv::imshow("wow", M);
		if (cv::waitKey(1) == 27) break;
	}


	ms = double(clock() - clk);
	std::cout << "Exec time:\t" << ms << " ms.\r\nRender Time:\t" << renderTime << "ms\r\nExpected FPS:\t" << 1000.00 / renderTime;


	cudaDeviceReset();

    return 0;
}

