
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
#include "RTRendererCUDARealization.h"
#include <QtWidgets/QApplication>
#include <QComboBox>

#include <QtCore/QDebug>
#include <QtGui/QImage>
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/opencv.hpp"

#define ALLOWOUTOFBOUND

clock_t StartTime;

unsigned char* ppm = new unsigned char[MAX_X * MAX_Y * 3 + 10000];

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


__global__ void converter(float* fBuffer, unsigned char* charBuffer, float upperBound)
{

	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;

	int index = ((MAX_Y - j - 1) * MAX_X  + i) * 3;

	charBuffer[index + 0] = clip(upperBound, 0.0, fBuffer[index + 2]) * (255.99 / upperBound);
	charBuffer[index + 1] = clip(upperBound, 0.0, fBuffer[index + 1]) * (255.99 / upperBound);
	charBuffer[index + 2] = clip(upperBound, 0.0, fBuffer[index + 0]) * (255.99 / upperBound);

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
	enableVTMode();
	QApplication a(argc, argv);
	RTRendererCUDAQT window;
	window.show();
	window.discardParameter();
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
	float renderTime;
	size_t* pValue = new size_t;
	cv::Mat M(MAX_Y, MAX_X, CV_32FC3, cv::Scalar(0, 0, 0));
	size_t frameBufferSize = 3 * MAX_X * MAX_Y * sizeof(float);
	float* frameBuffer;
	Hittable** cudaList;
	Hittable** cudaWorld;
	int num_Hittables = 500;
	float ms;
	curandState* worldGenRandState;
	dim3 blocks(MAX_X / BLK_X + 1, MAX_Y / BLK_Y + 1);
	dim3 threads(BLK_X, BLK_Y);
	curandState* renderRandomStates;
	Vec3 tLookat;
	Vec3 tLookfrom;
	Vec3 tVup;
	float renderStart;
	const char* fileName = "result.png";
	cv::Mat output;
	float tFocusDist;
	float tAperture;
	float tFov;
	int* allow = this->flag_show;
	unsigned char* charBuffer;

	while(this->end_flag == 0)
	{
		printMsg(LogLevel::info, "Rendering a %d x %d image in %d x %d blocks", MAX_X, MAX_Y, BLK_X, BLK_Y);
		printMsg(LogLevel::info, "SPP(per frame) = %d, depth = %d", SPP, ITER);
		printMsg(LogLevel::info, "Current log level: %d", logLevel);

	
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

	
		checkCudaErrors(cudaMallocManaged((void**)&frameBuffer, frameBufferSize));
		checkCudaErrors(cudaMalloc((void**)&charBuffer, sizeof(char)*(MAX_X * MAX_Y * 3 + 10000)))

		checkCudaErrors(cudaMalloc((void**)&cudaList, num_Hittables * sizeof(Hittable*)));
	
		checkCudaErrors(cudaMalloc((void**)&cudaWorld, sizeof(Hittable*)));
		checkCudaErrors(cudaMalloc((void**) & (this->cudaCam), sizeof(Camera*)));

		ms = float(clock() - StartTime);
		printMsg(LogLevel::info, "Alloc finished @ %lf ms", ms);

		checkCudaErrors(cudaMalloc((void**)&worldGenRandState, sizeof(curandState)));

		tLookat = *this->Lookat;
		tLookfrom = *this->Lookfrom;
		tVup = *this->Vup;
		tFocusDist = this->FocusDist;
		tAperture = this->Aperture;
		tFov = this->Fov;
	

		// createRandScene <<<1, 1 >>> (cudaList, cudaWorld, cudaCam, t, em.cols, em.rows, worldGenRandState, tLookat, tLookfrom,tVup,tFocusDist,tAperture,tFov);
		meshTestHost(cudaList, cudaWorld,allow, "lowpolydeer.obj");
		camInit <<<1, 1 >>> (tLookat, tLookfrom, tVup, tFocusDist, tAperture, tFov, this->cudaCam);
		

		checkCudaErrors(cudaGetLastError());
		checkCudaErrors(cudaDeviceSynchronize());

		ms = float(clock() - StartTime);
		printMsg(LogLevel::info, "World gen finished @ %lf ms", ms);


		checkCudaErrors(cudaMalloc((void**)&renderRandomStates, MAX_X * MAX_Y * sizeof(curandState)));
		rander_init << <blocks, threads >> > (renderRandomStates);
		checkCudaErrors(cudaGetLastError());
		checkCudaErrors(cudaDeviceSynchronize());

		ms = float(clock() - StartTime);
		renderStart = ms;
		printMsg(LogLevel::info, "Init renderer finished @ %lf ms", ms);

		printMsg(LogLevel::info, "\t+-------------------------------------------------------------------------------+");
		printMsg(LogLevel::info, "\t|                 Starting renderer, press q in prompt to quit.                 |");
		printMsg(LogLevel::info, "\t+---------------+---------------+---------------+---------------+---------------+");
		printMsg(LogLevel::info, "\t|      cur.     |      avg.     |     total     |      FPS      |      SPP      |");
		printMsg(LogLevel::info, "\t+---------------+---------------+---------------+---------------+---------------+");
		printMsg(LogLevel::info, "\t|               |               |               |               |               |");
		printMsg(LogLevel::info, "\t+---------------+---------------+---------------+---------------+---------------+\033[A\r");

		this->frameCount = 0;
		while (this->break_flag == 0)
		{
			if (pause_flag == 0)
			{

				renderTime = ms;

				renderer <<<blocks, threads >>> (this->frameCount++, frameBuffer, this->cudaCam, cudaWorld, renderRandomStates);
				converter <<<blocks, threads >>> (frameBuffer, charBuffer, this->targetClipUpperbound);

				checkCudaErrors(cudaGetLastError());
				checkCudaErrors(cudaDeviceSynchronize());
				checkCudaErrors(cudaMemcpy(ppm, charBuffer, sizeof(char) * (MAX_X * MAX_Y * 3 + 10000), cudaMemcpyDeviceToHost));

				ms = float(clock() - StartTime);
				renderTime = ms - renderTime;
				clearLine();
				printMsg(LogLevel::info, "\t|%*.2lf \t|%*.2lf \t|%*.2lf\t| %*.6lf\t| %*d\t|", 7, renderTime / 1000.0, 7, (ms - renderStart) / 1000.0 / this->frameCount, 7, (ms - renderStart) / 1000.0, 10, 1000.0 * this->frameCount / (ms - renderStart), 7, this->frameCount * SPP);


				emit refresh_flag();
			}


		}
		this->break_flag = 0;
	
		printMsg(LogLevel::info, "\t+---------------+---------------+---------------+---------------+");

		printMsg(LogLevel::debug, "Exec time: %lf ms. Saving result...", ms);


	
		M.data = (unsigned char *)frameBuffer;
		M *= 255.99;
		M.convertTo(output, CV_8UC3);
		cv::imwrite(fileName, output);
		printMsg(LogLevel::info, "File saved at: \"%s\"", fileName);
		checkCudaErrors(cudaFree(frameBuffer));
		checkCudaErrors(cudaFree(charBuffer));
		checkCudaErrors(cudaFree(cudaList));
		checkCudaErrors(cudaFree(cudaWorld));
		checkCudaErrors(cudaFree((this->cudaCam)));
		checkCudaErrors(cudaFree(worldGenRandState));
		checkCudaErrors(cudaFree(renderRandomStates));
	}
	
}


