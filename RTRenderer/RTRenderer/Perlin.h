#pragma once
#include "misc.h"

inline double trilinearInterpolate(Vec3 c[2][2][2], double u, double v, double w);

class Perlin
{
public:
	static Vec3* randVec;
	static int* xPermute;
	static int* yPermute;
	static int* zPermute;

	double noise(const Vec3& p) const
	{
		double u = p.x() - floor(p.x());
		double v = p.y() - floor(p.y());
		double w = p.z() - floor(p.z());

		int i = int(floor(p.x())) & 255;
		int j = int(floor(p.y())) & 255;
		int k = int(floor(p.z())) & 255;

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


	double turbulence(const Vec3& p, int depth = 7) const
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
};

static Vec3* generatePerlin()
{
	Vec3* p = new Vec3[256];
	for (int i = 0; i < 256; i++)
	{
		p[i] = Vec3(randD(), randD(), randD());
		p[i] *= 2;
		p[i] -= Vec3(1,1,1);
		p[i].makeUnitVector();
	}
	return p;
}

void permute(int* p, int n)
{
	for(int i = n - 1; i > 0; i--)
	{
		int target = int(randD() * (i + 1));
		int tmp = p[i];
		p[i] = p[target];
		p[target] = tmp;
	}
	return;
}

static int* perlinGeneratePermute()
{
	int* p = new int[256];
	for (int i = 0; i < 256; i++)
	{
		p[i] = i;
	}
	permute(p, 256);
	return p;
}

inline double trilinearInterpolate(Vec3 c[2][2][2], double u, double v, double w)
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

		Vec3 weight(u-i, v-j, w-k);

		accum += 
			(i * uu + (1 - i) * (1 - uu)) *
			(j * vv + (1 - j) * (1 - vv)) *
			(k * ww + (1 - k) * (1 - ww)) * 
			dot(c[i][j][k], weight);
	}
	//return (accum+1.0)*0.5;
	return accum;
}

Vec3* Perlin::randVec = generatePerlin();
int* Perlin::xPermute = perlinGeneratePermute();
int* Perlin::yPermute = perlinGeneratePermute();
int* Perlin::zPermute = perlinGeneratePermute();
