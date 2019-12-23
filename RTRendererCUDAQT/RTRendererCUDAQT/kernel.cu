
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



RTRendererCUDAQT::RTRendererCUDAQT(QWidget* parent)
	: QMainWindow(parent)
{
	ui.setupUi(this);
	setGeometry(10, 10, MAX_X + 400, MAX_Y + 400);

	Lab = new QLabel(this);
	Lab->setGeometry(20, 20, MAX_X, MAX_Y);

	logText = new QTextEdit(this);
	logText->setGeometry(MAX_X + 50, 20, 300, MAX_Y);

	StartButton = new QPushButton("Start", this);
	StartButton->setGeometry(MAX_X + 170, MAX_Y + 40, 100, 20);

	StopButton = new QPushButton("Stop", this);
	StopButton->setGeometry(MAX_X + 170, MAX_Y + 80, 100, 20);

	Updater = new QPushButton("Update", this);
	Updater->setGeometry(MAX_X + 170, MAX_Y + 120, 100, 20);

	Discarder = new QPushButton("Discard", this);
	Discarder->setGeometry(MAX_X + 170, MAX_Y + 160, 100, 20);

	CameraButton = new QPushButton("Camera", this);
	CameraButton->setGeometry(MAX_X + 50, MAX_Y + 40, 100, 20);

	WorldButton = new QPushButton("World", this);
	WorldButton->setGeometry(MAX_X + 50, MAX_Y + 80, 100, 20);

	RenderButton = new QPushButton("Render", this);
	RenderButton->setGeometry(MAX_X + 50, MAX_Y + 120, 100, 20);



	setLabelRender(0, 20, MAX_Y + 40, "TargetSPP");
	setLabelRender(1, 140, MAX_Y + 40, "ClipUpperbound");

	setLabelCamera(0, 20, MAX_Y + 40, "LookatX");
	setLabelCamera(1, 140, MAX_Y + 40, "LookatY");
	setLabelCamera(2, 260, MAX_Y + 40, "LookatZ");
	setLabelCamera(3, 20, MAX_Y + 80, "LookfromX");
	setLabelCamera(4, 140, MAX_Y + 80, "LookfromY");
	setLabelCamera(5, 260, MAX_Y + 80, "LookfromZ");
	setLabelCamera(6, 20, MAX_Y + 120, "VupX");
	setLabelCamera(7, 140, MAX_Y + 120, "VupY");
	setLabelCamera(8, 260, MAX_Y + 120, "VupZ");
	setLabelCamera(9, 20, MAX_Y + 160, "FocusDist");
	setLabelCamera(10, 140, MAX_Y + 160, "Aperture");
	setLabelCamera(11, 260, MAX_Y + 160, "Fov");

	setLabelWorld(0, 20, MAX_Y + 40, "PlaceHolder");

	looper = new LoopThread(this);

	connect(StartButton, SIGNAL(clicked()), this, SLOT(Startear()));
	connect(looper, SIGNAL(refresh_flag()), this, SLOT(refresh()));
	connect(StopButton, SIGNAL(clicked()), this, SLOT(Stop()));
	connect(Updater, SIGNAL(clicked()), this, SLOT(setParameter()));
	connect(Discarder, SIGNAL(clicked()), this, SLOT(discardParameter()));
	connect(CameraButton, SIGNAL(clicked()), this, SLOT(changeParaCamera()));
	connect(WorldButton, SIGNAL(clicked()), this, SLOT(changeParaWorld()));
	connect(RenderButton, SIGNAL(clicked()), this, SLOT(changeParaRender()));


	initialization();

}

void RTRendererCUDAQT::initialization()
{
	this->looper->break_flag = 0;
	this->looper->end_flag = 0;
	this->looper->targetClipUpperbound = 1.0;
	this->looper->targetSPP = INT_MAX;//默认是最大值，不依靠它提供break

	this->looper->Lookat = new Vec3(500, 500, 500);
	this->looper->Lookfrom = new Vec3(5000, 2000, 4000);
	this->looper->Vup = new Vec3(0, 1, 0);

	this->looper->FocusDist = (*this->looper->Lookat - *this->looper->Lookfrom).length();
	this->looper->Aperture = 0.05;
	this->looper->Fov = 50.0;

	discardParameter();
	hideAll();
}

void LoopThread::run()
{
	kernel();
}


//给自己加个模板，还有其他一些东西。
void LoopThread::PrintMessege()
{
	emit info_flag();
}

void LoopThread::checkBreak()
{
	if (this->frameCount * SPP >= this->targetSPP)
	{
		this->break_flag = 1;
	}
}

void RTRendererCUDAQT::setLabelCamera(int index, int x, int y, std::string name)
{
	labParaCamera[index] = new QLabel(this);
	lineParaCamera[index] = new  QLineEdit(this);
	labParaCamera[index]->setGeometry(x, y + 20, 100, 20);
	lineParaCamera[index]->setGeometry(x, y, 100, 20);
	labParaCamera[index]->setText(QString::fromStdString(name));
}

void RTRendererCUDAQT::setLabelRender(int index, int x, int y, std::string name)
{
	labParaRender[index] = new QLabel(this);
	lineParaRender[index] = new  QLineEdit(this);
	labParaRender[index]->setGeometry(x, y + 20, 100, 20);
	lineParaRender[index]->setGeometry(x, y, 100, 20);
	labParaRender[index]->setText(QString::fromStdString(name));
}

void RTRendererCUDAQT::setLabelWorld(int index, int x, int y, std::string name)
{
	labParaWorld[index] = new QLabel(this);
	lineParaWorld[index] = new  QLineEdit(this);
	labParaWorld[index]->setGeometry(x, y + 20, 100, 20);
	lineParaWorld[index]->setGeometry(x, y, 100, 20);
	labParaWorld[index]->setText(QString::fromStdString(name));
}




void RTRendererCUDAQT::hideAll()
{
	for (int i = 0; i < paraNumCamere; i++)
	{
		labParaCamera[i]->hide();
		lineParaCamera[i]->hide();
	}
	for (int i = 0; i < paraNumWorld; i++)
	{
		labParaWorld[i]->hide();
		lineParaWorld[i]->hide();
	}
	for (int i = 0; i < paraNumRender; i++)
	{
		labParaRender[i]->hide();
		lineParaRender[i]->hide();
	}
}

void RTRendererCUDAQT::changeParaCamera()
{
	changingNum = paraNumCamere;
	changingLab = labParaCamera;
	changingLine = lineParaCamera;
	changePara();
}

void RTRendererCUDAQT::changeParaWorld()
{
	changingNum = paraNumWorld;
	changingLab = labParaWorld;
	changingLine = lineParaWorld;
	changePara();
}

void RTRendererCUDAQT::changeParaRender()
{
	changingNum = paraNumRender;
	changingLab = labParaRender;
	changingLine = lineParaRender;
	changePara();
}

void RTRendererCUDAQT::changePara()
{
	hideAll();
	for (int i = 0; i < changingNum; i++)
	{
		changingLab[i]->show();
		changingLine[i]->show();
	}
}



void RTRendererCUDAQT::Startear()
{
	this->ShowPara();
	looper->start();
}

void RTRendererCUDAQT::Stop()
{
	this->looper->break_flag = 1;
	this->looper->end_flag = 1;
}

void RTRendererCUDAQT::setParameter()
{
	this->looper->targetSPP = this->lineParaRender[0]->text().toInt();
	this->looper->targetClipUpperbound = this->lineParaRender[1]->text().toFloat();

	this->looper->Lookat->e[0] = this->lineParaCamera[0]->text().toFloat();
	this->looper->Lookat->e[1] = this->lineParaCamera[1]->text().toFloat();
	this->looper->Lookat->e[2] = this->lineParaCamera[2]->text().toFloat();

	this->looper->Lookfrom->e[0] = this->lineParaCamera[3]->text().toFloat();
	this->looper->Lookfrom->e[1] = this->lineParaCamera[4]->text().toFloat();
	this->looper->Lookfrom->e[2] = this->lineParaCamera[5]->text().toFloat();

	this->looper->Vup->e[0] = this->lineParaCamera[6]->text().toFloat();
	this->looper->Vup->e[1] = this->lineParaCamera[7]->text().toFloat();
	this->looper->Vup->e[2] = this->lineParaCamera[8]->text().toFloat();

	//手动对焦  有输入参数
	if (this->lineParaCamera[9]->text().toFloat() != this->looper->FocusDist)
		this->looper->FocusDist = this->lineParaCamera[9]->text().toFloat();
	else
		this->looper->FocusDist = (*this->looper->Lookat - *this->looper->Lookfrom).length();
	
	
	this->looper->Aperture = this->lineParaCamera[10]->text().toFloat();
	this->looper->Fov = this->lineParaCamera[11]->text().toFloat();

	this->looper->placeHolder = this->lineParaWorld[0]->text().toFloat();

	this->ShowPara();
	this->looper->frameCount = 0;
	this->looper->break_flag = 1;

}

void RTRendererCUDAQT::ShowPara()
{
	std::string str = "SPP Threshold =" + std::to_string(this->looper->targetSPP);
	this->logText->append(QString::fromStdString(str));
	str = "Clip Upperbound =" + std::to_string(this->looper->targetClipUpperbound);
	this->logText->append(QString::fromStdString(str));

	str = "LookatX =" + std::to_string(this->looper->Lookat->e[0]);
	this->logText->append(QString::fromStdString(str));
	str = "LookatY =" + std::to_string(this->looper->Lookat->e[1]);
	this->logText->append(QString::fromStdString(str));
	str = "LookatZ =" + std::to_string(this->looper->Lookat->e[2]);
	this->logText->append(QString::fromStdString(str));
	str = "LookfromX =" + std::to_string(this->looper->Lookfrom->e[0]);
	this->logText->append(QString::fromStdString(str));
	str = "LookfromY =" + std::to_string(this->looper->Lookfrom->e[1]);
	this->logText->append(QString::fromStdString(str));
	str = "LookfromZ =" + std::to_string(this->looper->Lookfrom->e[2]);
	this->logText->append(QString::fromStdString(str));
	str = "VupX =" + std::to_string(this->looper->Vup->e[0]);
	this->logText->append(QString::fromStdString(str));
	str = "VupY =" + std::to_string(this->looper->Vup->e[1]);
	this->logText->append(QString::fromStdString(str));
	str = "VupZ =" + std::to_string(this->looper->Vup->e[2]);
	this->logText->append(QString::fromStdString(str));
	str = "FocusDist =" + std::to_string(this->looper->FocusDist);
	this->logText->append(QString::fromStdString(str));
	str = "Aperture =" + std::to_string(this->looper->Aperture);
	this->logText->append(QString::fromStdString(str));
	str = "Fov =" + std::to_string(this->looper->Fov);
	this->logText->append(QString::fromStdString(str));

	str = "Place Holder =" + std::to_string(this->looper->placeHolder);
	this->logText->append(QString::fromStdString(str));

	this->discardParameter();

}

void RTRendererCUDAQT::discardParameter()
{
	this->lineParaRender[0]->setText(QString::fromStdString(std::to_string(this->looper->targetSPP)));
	this->lineParaRender[1]->setText(QString::fromStdString(std::to_string(this->looper->targetClipUpperbound)));

	this->lineParaCamera[0]->setText(QString::fromStdString(std::to_string(this->looper->Lookat->e[0])));
	this->lineParaCamera[1]->setText(QString::fromStdString(std::to_string(this->looper->Lookat->e[1])));
	this->lineParaCamera[2]->setText(QString::fromStdString(std::to_string(this->looper->Lookat->e[2])));
	this->lineParaCamera[3]->setText(QString::fromStdString(std::to_string(this->looper->Lookfrom->e[0])));
	this->lineParaCamera[4]->setText(QString::fromStdString(std::to_string(this->looper->Lookfrom->e[1])));
	this->lineParaCamera[5]->setText(QString::fromStdString(std::to_string(this->looper->Lookfrom->e[2])));
	this->lineParaCamera[6]->setText(QString::fromStdString(std::to_string(this->looper->Vup->e[0])));
	this->lineParaCamera[7]->setText(QString::fromStdString(std::to_string(this->looper->Vup->e[1])));
	this->lineParaCamera[8]->setText(QString::fromStdString(std::to_string(this->looper->Vup->e[2])));
	this->lineParaCamera[9]->setText(QString::fromStdString(std::to_string(this->looper->FocusDist)));
	this->lineParaCamera[10]->setText(QString::fromStdString(std::to_string(this->looper->Aperture)));
	this->lineParaCamera[11]->setText(QString::fromStdString(std::to_string(this->looper->Fov)));

	this->lineParaWorld[0]->setText(QString::fromStdString(std::to_string(this->looper->placeHolder)));
}






#define ALLOWOUTOFBOUND

clock_t StartTime;

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


unsigned char* ppm = new unsigned char [MAX_X * MAX_Y * 3 + 10000];


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
	cv::Mat em;
	unsigned char* t;
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
	checkCudaErrors(cudaMalloc((void**)&cudaList, num_Hittables * sizeof(Hittable*)));
	
	checkCudaErrors(cudaMalloc((void**)&cudaWorld, sizeof(Hittable*)));
	checkCudaErrors(cudaMalloc((void**) & (this->cudaCam), sizeof(Camera*)));

	ms = float(clock() - StartTime);
	printMsg(LogLevel::info, "Alloc finished @ %lf ms", ms);

	checkCudaErrors(cudaMalloc((void**)&worldGenRandState, sizeof(curandState)));

	em = cv::imread("earthmap.jpg");
	if (em.rows < 1 || em.cols < 1)
	{
		printMsg(LogLevel::error, "Failed to find Earth texture(earthmap.jpg).");
	}
	else
	{
		printMsg(LogLevel::debug, "Texture loaded.");
	}
	
	checkCudaErrors(cudaMalloc((void**)&t, sizeof(unsigned char) * em.rows * em.cols * 3));
	checkCudaErrors(cudaMemcpy(t, em.data, sizeof(unsigned char) * em.rows * em.cols * 3, cudaMemcpyHostToDevice));

	tLookat = *this->Lookat;
	tLookfrom = *this->Lookfrom;
	tVup = *this->Vup;
	tFocusDist = this->FocusDist;
	tAperture = this->Aperture;
	tFov = this->Fov;

	// createRandScene <<<1, 1 >>> (cudaList, cudaWorld, cudaCam, t, em.cols, em.rows, worldGenRandState, tLookat, tLookfrom,tVup,tFocusDist,tAperture,tFov);
	meshTestHost(t, em.cols, em.rows, cudaList, cudaWorld);
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
		renderTime = ms;

		renderer << <blocks, threads >> > (this->frameCount++, frameBuffer, this->cudaCam, cudaWorld, renderRandomStates);


		checkCudaErrors(cudaGetLastError());
		checkCudaErrors(cudaDeviceSynchronize());
		ms = float(clock() - StartTime);
		renderTime = ms - renderTime;
		clearLine();
		printMsg(LogLevel::info, "\t|%*.2lf \t|%*.2lf \t|%*.2lf\t| %*.6lf\t| %*d\t|", 7, renderTime / 1000.0, 7, (ms - renderStart) / 1000.0 / this->frameCount, 7, (ms - renderStart) / 1000.0, 10, 1000.0 * this->frameCount / (ms - renderStart), 7, this->frameCount * SPP);


		for (int i = 0; i < MAX_Y; i++)
		{
			for (int j = 0; j < MAX_X; j++)
			{
				int index = 3 * (i * MAX_X + j);
				for (int k = 0; k < 3; k++)
				{
					ppm[index + k] = clip(this->targetClipUpperbound, 0.0, frameBuffer[index + 2 - k]) * (255.99 / this->targetClipUpperbound);
				}
			}
		}

		emit refresh_flag();

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
	checkCudaErrors(cudaFree(cudaList));
	checkCudaErrors(cudaFree(cudaWorld));
	checkCudaErrors(cudaFree((this->cudaCam)));
	checkCudaErrors(cudaFree(worldGenRandState));
	checkCudaErrors(cudaFree(t));
	checkCudaErrors(cudaFree(renderRandomStates));
	
	}
	
}


