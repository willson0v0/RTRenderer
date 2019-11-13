#pragma once
#include "misc.h"
#include "Vec3.h"

class Texture
{
public:
	__device__ virtual Vec3 value(double u, double v, const Vec3& p) const = 0;
};

class ConstantTexture : public Texture
{
public:
	Vec3 color;

	__device__ ConstantTexture() {}
	__device__ ConstantTexture(Vec3 c) : color(c) {}
	__device__ ConstantTexture(double r, double g, double b) : color(Vec3(r, g ,b)) {}
	__device__ virtual Vec3 value(double u, double v, const Vec3& p) const
	{
		return color;
	}
};

class CheckerTexture : public Texture
{
public:
	Texture* odd;
	Texture* even;

	__device__ CheckerTexture() :even(new ConstantTexture(1, 1, 1)), odd(new ConstantTexture(0, 0, 0)) {}
	__device__ CheckerTexture(Texture* t0, Texture* t1) :even(t0), odd(t1) {}

	__device__ virtual Vec3 value(double u, double v, const Vec3& p) const
	{
		double sines = sin(10 * p.e[0]) * sin(10 * p.e[1]) * sin(10 * p.e[2]);
		if (sines < 0)
			return odd->value(u, v, p);
		else
			return even->value(u, v, p);
	}
};

class ImageTexture : public Texture
{
public:
	unsigned char* data;
	int imgX, imgY;

	__device__ ImageTexture(unsigned char* d, int x, int y)
		:data(d), imgX(x), imgY(y) {}

	__device__ virtual Vec3 value(double u, double v, const Vec3& p) const
	{
		int i = u * imgX;
		int j = (1 - v) * imgY;
		i = i < 0 ? 0 : (i > imgX - 1 ? imgX - 1 : i);
		j = j < 0 ? 0 : (j > imgY - 1 ? imgY - 1 : j);

		return Vec3(
			data[j * imgX * 3 + i * 3 + 2] / 255.0,
			data[j * imgX * 3 + i * 3 + 1] / 255.0,
			data[j * imgX * 3 + i * 3 + 0] / 255.0);
	}
};