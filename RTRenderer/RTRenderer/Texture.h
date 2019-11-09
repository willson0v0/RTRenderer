#pragma once

#include "Vec3.h"
#include "Perlin.h"
#include "opencv/cv.hpp"

class Texture
{
public:
	virtual Vec3 value(double u, double v, const Vec3& p) const = 0;
};

class constantTexture : public Texture
{
public:
	Vec3 color;

	constantTexture() {}
	constantTexture(Vec3 c) : color(c) {}
	constantTexture(double r, double g, double b) : color(Vec3(r, g, b)) {}
	virtual Vec3 value(double u, double v, const Vec3& p) const
	{
		return color;
	}
};

class CheckerTexture : public Texture
{
public:
	Texture* odd;
	Texture* even;

	CheckerTexture() :even(new constantTexture(1, 1, 1)), odd(new constantTexture(0, 0, 0)) {}
	CheckerTexture(Texture* t0, Texture* t1) :even(t0), odd(t1) {}

	virtual Vec3 value(double u, double v, const Vec3& p) const
	{
		double sines = sin(10 * p.x()) * sin(10 * p.y()) * sin(10 * p.z());
		if (sines < 0)
			return odd->value(u, v, p);
		else
			return even->value(u, v, p);
	}
};

class NoiseTexture : public Texture
{
public:
	Perlin noise;
	double scale;

	NoiseTexture() :scale(1) {}
	NoiseTexture(double sc) :scale(sc) {}
	virtual Vec3 value(double u, double v, const Vec3& p) const
	{
		// return Vec3(1, 1, 1) * noise.turbulence(p * 4);
		return Vec3(1, 1, 1) * 0.5 * (1 + sin(scale * p.z() + 10 * noise.turbulence(p)));
	}
};

class ImageTexture : public Texture
{
public:
	cv::Mat data;
	int imgX, imgY;

	ImageTexture(cv::Mat d, int x, int y)
		:data(d), imgX(x), imgY(y) {}

	virtual Vec3 value(double u, double v, const Vec3& p) const
	{
		int i = u * imgX;
		int j = (1-v) * imgY;
		i = i < 0 ? 0 : (i > imgX - 1 ? imgX - 1 : i);
		j = j < 0 ? 0 : (j > imgY - 1 ? imgY - 1 : j);
		return Vec3b2Vec3(data.at<cv::Vec3b>(j, i));
	}
};