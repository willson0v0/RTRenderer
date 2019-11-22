#pragma once

#include "misc.h"

__device__ inline double trilinearInterpolate(Vec3 c[2][2][2], double u, double v, double w);

__device__ Vec3* randVec;
__device__ int* xPermute;
__device__ int* yPermute;
__device__ int* zPermute;


#define RND (curand_uniform(localRandState))
__device__ static Vec3* generatePerlin(curandState* localRandState)
{
	Vec3* p = new Vec3[256];
	for (int i = 0; i < 256; i++)
	{
		p[i] = Vec3(RND, RND, RND);
		p[i] *= 2;
		p[i] -= Vec3(1, 1, 1);
		p[i].makeUnitVector();
	}

	return p;
}

__device__ void permute(int* p, int n, curandState* localRandState)
{
	for (int i = n - 1; i > 0; i--)
	{
		int target = int(RND * (i + 1));
		int tmp = p[i];
		p[i] = p[target];
		p[target] = tmp;
	}
	return;
}

__device__ static int* perlinGeneratePermute(curandState* localRandState)
{
	int* p = new int[256];
	for (int i = 0; i < 256; i++)
	{
		p[i] = i;
	}
	permute(p, 256, localRandState);
	return p;
}

__device__ inline double trilinearInterpolate(Vec3 c[2][2][2], double u, double v, double w)
{
	double accum = 0;
	double uu, vv, ww;
	uu = u * u * (3 - 2 * u);   // Hermite cubit, eliminate Mach band
	vv = v * v * (3 - 2 * v);
	ww = w * w * (3 - 2 * w);

	for (int loop = 0; loop < 8; loop++)
	{
		int i = loop & 0b001;
		int j = (loop & 0b010) >> 1;
		int k = (loop & 0b100) >> 2;

		Vec3 weight(u - i, v - j, w - k);

		accum +=
			(i * uu + (1 - i) * (1 - uu)) *
			(j * vv + (1 - j) * (1 - vv)) *
			(k * ww + (1 - k) * (1 - ww)) *
			dot(c[i][j][k], weight);
	}
	//return (accum+1.0)*0.5;
	return accum;
}

class Perlin
{
public:
	__device__ double noise(const Vec3& p) const
	{
		double u = p.e[0] - floor(p.e[0]);
		double v = p.e[1] - floor(p.e[1]);
		double w = p.e[2] - floor(p.e[2]);

		int i = int(floor(p.e[0])) & 255;
		int j = int(floor(p.e[1])) & 255;
		int k = int(floor(p.e[2])) & 255;

		Vec3 c[2][2][2];

#pragma omp parallel for
		for (int loop = 0; loop < 8; loop++)
		{
			int di = loop & 0b001;
			int dj = (loop & 0b010) >> 1;
			int dk = (loop & 0b100) >> 2;

			c[di][dj][dk] = randVec[
				xPermute[(i + di) & 255] ^
				yPermute[(j + dj) & 255] ^
				zPermute[(k + dk) & 255]
			];
		}

		return trilinearInterpolate(c, u, v, w);
	}


	__device__ double turbulence(const Vec3& p, int depth = 7) const
	{
		double accum = 0;
		Vec3 tp = p;
		double weight = 1;
		for (int i = 0; i < depth; i++)
		{
			accum += weight * noise(tp);
			weight *= 0.5;
			tp *= 2;
		}
		return fabs(accum);
	}

	__device__ static void initPerlin(curandState* localRandState)
	{
		curandState rs = *localRandState;
		randVec = generatePerlin(localRandState);
		xPermute = perlinGeneratePermute(localRandState);
		yPermute = perlinGeneratePermute(localRandState);
		zPermute = perlinGeneratePermute(localRandState);
		*localRandState = rs;
	}
};