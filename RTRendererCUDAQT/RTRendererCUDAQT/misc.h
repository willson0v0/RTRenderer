#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <curand_kernel.h>
#include <iostream>
#include <math.h>
#include <functional>
#include <random>
#include <opencv/cv.hpp>
#include <stdio.h>
#include <stdarg.h>
#include <Windows.h>
#include <string>

std::string str;


#include "Vec3.h"

__device__ __host__ float ffmin(float a, float b)
{
	return a < b ? a : b;
}
__device__ __host__ float ffmax(float a, float b)
{
	return a > b ? a : b;
}

//											func_name		  line no
//												v				 v
#define checkCudaErrors(val) checkCuda((val), #val, __FILE__, __LINE__);
//										^				^
//									  func			file name

__device__ void getSphereUV(const Vec3& p, float& u, float& v)
{
	float phi = atan2(p.e[2], p.e[0]);
	float theta = asin(p.e[1]);
	u = 1 - (phi + PI) / (2 * PI);
	v = (theta + PI / 2) / PI;
}

__device__ Vec3 randomVecInUnitSphere(curandState* localRandState)
{
	Vec3 p;
	do
	{
		p = 2 * Vec3(curand_uniform(localRandState), curand_uniform(localRandState), curand_uniform(localRandState)) - Vec3(1,1,1);
	} while (p.squaredLength() >= 1);
	return p;
}

__device__ Vec3 randomVecInUnitDisk(curandState* localRandState)
{
	Vec3 p;
	do
	{
		p = 2 * Vec3(curand_uniform(localRandState), curand_uniform(localRandState), 0) - Vec3(1, 1, 0);
	} while (p.squaredLength() >= 1);
	return p;
}

__device__ Vec3 reflect(const Vec3& v, const Vec3& norm)
{
	return v - norm * (2 * dot(v, norm));
}

__device__ bool refract(const Vec3& v, const Vec3& n, float rri, Vec3& refracted) // rri: relative refractive index.
{
	Vec3 uv = unitVector(v);
	float dt = dot(uv, n);  //in angle
	float dis = 1 - rri * rri * (1 - dt * dt); // is total internal reflection?
	if (dis > 0)
	{
		refracted = rri * (uv - n * dt) - n * sqrt(dis);
		return true;
	}
	return false;
}

__device__ float schlick(float cosine, float refIndex) {
	float r0 = (1 - refIndex) / (1 + refIndex);
	r0 = r0 * r0;
	return r0 + (1 - r0) * pow((1 - cosine), 5);
}

__host__ inline float randD()
{
	static std::uniform_real_distribution<float> distribution(0.0, 1.0);
	static std::mt19937 generator;
	static std::function<float()> rand_generator =
		std::bind(distribution, generator);
	return rand_generator();
}

//将这个函数变成GUI的
//到时候照实说，LGZ写了函数原型，我将它改到GUI上

template<typename... Arguments>
__host__ __device__ void printMsg(LogLevel ll, const char* msg, Arguments... args)
{
#ifndef __CUDA_ARCH__
	float currentTimeMs = clock() - StartTime;
#else
	float currentTimeMs = NAN;
#endif // !__CUDA_ARCH__

	switch (ll)
	{
	case LogLevel::extra:
		if (logLevel >= LogLevel::extra)
		{
			if (VTModeEnabled) printf(ANSI_COLOR_CYAN);
			printf("[ Extra\t] %*.2lf\t: ", 6, currentTimeMs / 1000.0);
			printf(msg, args...);
			printf("\n");
		}
		break;
	case LogLevel::debug:
		if (logLevel >= LogLevel::debug)
		{
			if (VTModeEnabled) printf(ANSI_COLOR_GREEN);
			printf("[ Debug\t] %*.2lf\t: ", 6, currentTimeMs / 1000.0);
			printf(msg, args...);
			printf("\n");
		}
		break;
	case LogLevel::info:
		if (logLevel >= LogLevel::info)
		{
			if (VTModeEnabled) printf(ANSI_COLOR_RESET);
			printf("[ Info\t] %*.2lf\t: ", 6, currentTimeMs / 1000.0);
			printf(msg, args...);
			printf("\n");
		}
		break;
	case LogLevel::warning:
		if (logLevel >= LogLevel::warning)
		{
			if (VTModeEnabled) printf(ANSI_COLOR_YELLOW);
			printf("[Warning] %*.2lf\t: ", 6, currentTimeMs / 1000.0);
			printf(msg, args...);
			printf("\n");
		}
		break;
	case LogLevel::error:
		if (logLevel >= LogLevel::error)
		{
			if (VTModeEnabled) printf(ANSI_COLOR_RED);
			printf("[ Error\t] %*.2lf\t: ", 6, currentTimeMs / 1000.0);
			printf(msg, args...);
			printf("\n");
		}
		break;
	case LogLevel::fatal:
		if (logLevel >= LogLevel::fatal)
		{
			if (VTModeEnabled) printf(ANSI_COLOR_RED);
			printf("[ Fatal\t] %*.2lf\t: ", 6, currentTimeMs / 1000.0);
			printf(msg, args...);
			printf("\n");
		}
		break;
	default:
		break;
	}
	if (VTModeEnabled) printf(ANSI_COLOR_RESET);
}







__host__ __device__ void clearLine()
{
	printf("\033[A\033[2K\r");
}

__host__ bool enableVTMode()
{
	HANDLE hOut = GetStdHandle(STD_OUTPUT_HANDLE);
	if (hOut == INVALID_HANDLE_VALUE)
	{
		printMsg(LogLevel::error, "Failed to get STD_OUTPUT_HANDLE. ANSI color wont behave properly.");
		return false;
	}

	DWORD dwMode = 0;
	if (!GetConsoleMode(hOut, &dwMode))
	{
		printMsg(LogLevel::error, "Failed to get current CMD properties. ANSI color wont behave properly.");
		return false;
	}

	dwMode |= ENABLE_VIRTUAL_TERMINAL_PROCESSING;
	if (!SetConsoleMode(hOut, dwMode))
	{
		printMsg(LogLevel::error, "Failed to set CMD to VT mode. ANSI color wont behave properly.");
		return false;
	}
	printMsg(LogLevel::info, "Set CMD to VT mode.");
	VTModeEnabled = true;
	return true;
}

// DON'T CALL THIS! Use marco to auto generate msgs.
__host__ void checkCuda(cudaError_t result, char const* const func, const char* const file, int const line)
{
	if (result)
	{
		printMsg(LogLevel::fatal, "Cuda Error: \n %s (%s) @ %s: %d, %s\n",
			cudaGetErrorName(result),
			cudaGetErrorString(result),
			file,
			line,
			func);
		cudaDeviceReset();
		system("pause");
		exit(-1);
	}
}

__device__ __host__ float clip(float upperBound, float loweBound, float in)
{
	return in > upperBound ? upperBound : (in < loweBound ? loweBound : in);
}