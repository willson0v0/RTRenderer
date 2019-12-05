
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <iomanip>
#include <conio.h>
#include <opencv/cv.hpp>
#include <opencv2/photo/cuda.hpp>
#include "cudaExamp.h"
#include "cuda.h"
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
#include <thread>
#include <chrono>
#include <mutex>

#include "RTRendererCUDAQT.h"
#include <QtWidgets/QApplication>



#include <QtCore/QDebug>
#include <QtGui/QImage>
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/opencv.hpp"


#define ALLOWOUTOFBOUND

constexpr auto ITER = 50;
constexpr auto SPP = 4;


__device__ Vec3 color(const Ray& r, Hittable** world, int depth, curandState* localRandState)
{
	HitRecord rec;
	if ((*world)->hit(r, 0.001, FLT_MAX, rec, localRandState)) {
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
		Vec3 unit_direction = unitVector(r.direction);
		float t = 0.5f * (unit_direction.e[1] + 1.0f);
		Vec3 c = (1.0f - t) * Vec3(1.0, 1.0, 1.0) + t * Vec3(0.5, 0.7, 1.0);
#endif
		return c;
	}
}

// Main rander func.
__global__ void renderer(int frameCount, float* fBuffer, Camera** cam, Hittable** world, curandState* randState)  //{b, g, r}, stupid opencv
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
		float u = float(i + curand_uniform(&localRandState)) / float(MAX_X);
		float v = float(j + curand_uniform(&localRandState)) / float(MAX_Y);
		Ray r = (*cam)->getRay(u, v, &localRandState);
		pixel += color(r, world, 0 ,&localRandState);
	}
	randState[index] = localRandState;
	pixel /= float(SPP);
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

void func()
{
	;
}

unsigned char* ppm = new unsigned char [MAX_X * MAX_Y * 3 + 10000];



int main(int argc, char* argv[])
{
	enableVTMode();
	QApplication a(argc, argv);
	RTRendererCUDAQT window;
	window.show();
	a.exec();
}

void RTRendererCUDAQT::refresh()
{
	QImage image(ppm, MAX_X, MAX_Y, MAX_X * 3, QImage::Format_RGB888);
	image.rgbSwapped();
	Lab->clear();
	Lab->setPixmap(QPixmap::fromImage(image));
	Lab->repaint();
}


void LoopThread::kernel()
{
	func();
	float renderTime;

	printMsg(LogLevel::info, "Rendering a %d x %d image in %d x %d blocks", MAX_X, MAX_Y, BLK_X, BLK_Y);
	printMsg(LogLevel::info, "SPP(per frame) = %d, depth = %d", SPP, ITER);
	printMsg(LogLevel::info, "Current log level: %d", logLevel);

	size_t* pValue = new size_t;
	checkCudaErrors(cudaDeviceGetLimit(pValue, cudaLimitStackSize));
	printMsg(LogLevel::debug, "Stack size limit: \t\t%zu Byte.", *pValue);

	if (*pValue < 65536)
	{
		printMsg(LogLevel::warning, "Stack size too small(%zu Byte), Resizing to 65536...", *pValue);
		checkCudaErrors(cudaDeviceSetLimit(cudaLimitStackSize, 65536));
		checkCudaErrors(cudaDeviceGetLimit(pValue, cudaLimitStackSize));
		printMsg(LogLevel::debug, "Stack size limit: \t\t%zu Byte.", *pValue);
		if (*pValue < 65535)
		{
			printMsg(LogLevel::fatal, "Stack resized failed. Quit now.");
			cudaDeviceReset();
			system("pause");
			exit(-1);
		}
	}

	checkCudaErrors(cudaDeviceGetLimit(pValue, cudaLimitPrintfFifoSize));
	printMsg(LogLevel::debug, "Printf FIFO size: \t\t%zu Byte.", *pValue);
	checkCudaErrors(cudaDeviceGetLimit(pValue, cudaLimitMallocHeapSize));
	printMsg(LogLevel::debug, "Heap size limit: \t\t%zu Byte.", *pValue);
	checkCudaErrors(cudaDeviceGetLimit(pValue, cudaLimitDevRuntimeSyncDepth));
	printMsg(LogLevel::debug, "DevRuntimeSyncDepth: \t\t%zu .", *pValue);
	checkCudaErrors(cudaDeviceGetLimit(pValue, cudaLimitDevRuntimePendingLaunchCount));
	printMsg(LogLevel::debug, "DevRuntimePendingLaunchCount: %zu Byte.", *pValue);
	checkCudaErrors(cudaDeviceGetLimit(pValue, cudaLimitMaxL2FetchGranularity));
	printMsg(LogLevel::debug, "MaxL2FetchGranularity: \t%zu Byte.", *pValue);

#ifdef _DEBUG
	printMsg(LogLevel::warning, "Compiled under debug mode. Performance is compromised.");
#endif

	cv::Mat M(MAX_Y, MAX_X, CV_32FC3, cv::Scalar(0, 0, 0));

	size_t frameBufferSize = 3 * MAX_X * MAX_Y * sizeof(float);
	float* frameBuffer;
	checkCudaErrors(cudaMallocManaged((void**)&frameBuffer, frameBufferSize));

	Hittable** cudaList;
	int num_Hittables = 500;
	checkCudaErrors(cudaMalloc((void**)&cudaList, num_Hittables * sizeof(Hittable*)));
	Hittable** cudaWorld;
	checkCudaErrors(cudaMalloc((void**)&cudaWorld, sizeof(Hittable*)));
	Camera** cudaCam;
	checkCudaErrors(cudaMalloc((void**)&cudaCam, sizeof(Camera*)));

	float ms = float(clock() - StartTime);
	printMsg(LogLevel::info, "Alloc finished @ %lf ms", ms);

	curandState* worldGenRandState;
	checkCudaErrors(cudaMalloc((void**)&worldGenRandState, sizeof(curandState)));

	cv::Mat em = cv::imread("earthmap.jpg");
	if (em.rows < 1 || em.cols < 1)
	{
		printMsg(LogLevel::error, "Failed to find Earth texture(earthmap.jpg).");
	}
	else
	{
		printMsg(LogLevel::debug, "Texture loaded.");
	}
	unsigned char* t;
	checkCudaErrors(cudaMalloc((void**)&t, sizeof(unsigned char) * em.rows * em.cols * 3));
	checkCudaErrors(cudaMemcpy(t, em.data, sizeof(unsigned char) * em.rows * em.cols * 3, cudaMemcpyHostToDevice));

	createRandScene <<<1, 1 >>> (cudaList, cudaWorld, cudaCam, t, em.cols, em.rows, worldGenRandState);
	// createWorld1 <<<1, 1 >>> (cudaList, cudaWorld, cudaCam, worldGenRandState);
	// createCheckerTest <<<1, 1 >>> (cudaList, cudaWorld, cudaCam, worldGenRandState);
	// createCornellBox <<<1, 1 >>> (cudaList, cudaWorld, cudaCam, worldGenRandState);
	// createCornellSmoke <<<1, 1 >>> (cudaList, cudaWorld, cudaCam, worldGenRandState);

	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	ms = float(clock() - StartTime);
	printMsg(LogLevel::info, "World gen finished @ %lf ms", ms);

	dim3 blocks(MAX_X / BLK_X + 1, MAX_Y / BLK_Y + 1);
	dim3 threads(BLK_X, BLK_Y);

	curandState* renderRandomStates;
	checkCudaErrors(cudaMalloc((void**)&renderRandomStates, MAX_X * MAX_Y * sizeof(curandState)));
	rander_init <<<blocks, threads >>> (renderRandomStates);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	ms = float(clock() - StartTime);
	float renderStart = ms;
	printMsg(LogLevel::info, "Init renderer finished @ %lf ms", ms);

	printMsg(LogLevel::info, "\t+-------------------------------------------------------------------------------+");
	printMsg(LogLevel::info, "\t|                 Starting renderer, press q in prompt to quit.                 |");
	printMsg(LogLevel::info, "\t+---------------+---------------+---------------+---------------+---------------+");
	printMsg(LogLevel::info, "\t|      cur.     |      avg.     |     total     |      FPS      |      SPP      |");
	printMsg(LogLevel::info, "\t+---------------+---------------+---------------+---------------+---------------+");
	printMsg(LogLevel::info, "\t|               |               |               |               |               |");
	printMsg(LogLevel::info, "\t+---------------+---------------+---------------+---------------+---------------+\033[A\r");

	int frameCount = 0;
	int break_ctrl = 1;
	while (1)
	{
		renderTime = ms;

		renderer <<<blocks, threads >>> (frameCount++, frameBuffer, cudaCam, cudaWorld, renderRandomStates);


		checkCudaErrors(cudaGetLastError());
		checkCudaErrors(cudaDeviceSynchronize());

		ms = float(clock() - StartTime);
		renderTime = ms - renderTime;
		clearLine();
		printMsg(LogLevel::info, "\t|%*.2lf \t|%*.2lf \t|%*.2lf\t| %*.6lf\t| %*d\t|", 7, renderTime / 1000.0, 7, (ms - renderStart) / 1000.0 / frameCount, 7, (ms - renderStart) / 1000.0, 10, 1000.0 * frameCount / (ms - renderStart),7, frameCount * SPP);

		/*
		M.data = (uchar*)frameBuffer;
		cv::imshow("wow", M);
		if (cv::waitKey(1) == 27) break;;
		*/
		
		/*
		for (int i = 0; i < 3 * MAX_X * MAX_Y; i++)
		{
			if (frameBuffer[i] >= 1)
				ppm[i] = 255;
			else
				ppm[i] = frameBuffer[i] * 255.99;
		}
		*/

		for (int i = 0; i < MAX_Y; i++)
		{
			for (int j = 0; j < MAX_X; j++)
			{
				int index = 3 * (i * MAX_X + j);
				for (int k = 0; k < 3; k++)
				{
					ppm[index + k] = clip(1.0, 0.0, frameBuffer[index + 2 - k]) * 255.99;
				}
			}
		}
		
		emit refresh_flag();

		if (this->break_flag == 1)
		{
			this->break_flag = 0;
			break;
		}

		if (break_ctrl == this->pre_break)
		{
			break;
		}
		else
		{
			break_ctrl++;
		}
	}
	printMsg(LogLevel::info, "\t+---------------+---------------+---------------+---------------+");

	printMsg(LogLevel::debug, "Exec time: %lf ms. Saving result...", ms);

	const char* fileName = "result.png";
	cv::Mat output;
	M *= 255.99;
	M.convertTo(output, CV_8UC3);
	cv::imwrite(fileName, output);

	printMsg(LogLevel::info, "File saved at: \"%s\"", fileName);
	checkCudaErrors(cudaDeviceReset());
	printMsg(LogLevel::debug, "Device reset finished.");
}


