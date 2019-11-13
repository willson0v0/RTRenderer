#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <curand_kernel.h>
#include <iostream>
#include <math.h>
#include <functional>
#include <random>
#include "Vec3.h"
#include <opencv/cv.hpp>

//											func_name		  line no
//												v				 v
#define checkCudaErrors(val) checkCuda((val), #val, __FILE__, __LINE__);
//										^				^
//									  func			file name

// DON'T CALL THIS! Use marco to auto generate msgs.
void checkCuda(cudaError_t result, char const* const func, const char* const file, int const line)
{
	if (result)
	{
		std::cerr 
			<< " CUDA ERROR: \r\n" 
			<< cudaGetErrorName(result) <<" : "<<cudaGetErrorString(result)
			<< " @ " << file 
			<< " : " << line 
			<< " , " << func << std::endl;
		cudaDeviceReset();
		system("pause");
		exit(-1);
	}
}

__device__ void getSphereUV(const Vec3& p, double& u, double& v)
{
	double phi = atan2(p.e[2], p.e[0]);
	double theta = asin(p.e[1]);
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

__device__ bool refract(const Vec3& v, const Vec3& n, double rri, Vec3& refracted) // rri: relative refractive index.
{
	Vec3 uv = unitVector(v);
	double dt = dot(uv, n);  //in angle
	double dis = 1 - rri * rri * (1 - dt * dt); // is total internal reflection?
	if (dis > 0)
	{
		refracted = rri * (uv - n * dt) - n * sqrt(dis);
		return true;
	}
	return false;
}

__device__ double schlick(double cosine, double refIndex) {
	double r0 = (1 - refIndex) / (1 + refIndex);
	r0 = r0 * r0;
	return r0 + (1 - r0) * pow((1 - cosine), 5);
}

__host__ inline double randD()
{
	static std::uniform_real_distribution<double> distribution(0.0, 1.0);
	static std::mt19937 generator;
	static std::function<double()> rand_generator =
		std::bind(distribution, generator);
	return rand_generator();
}