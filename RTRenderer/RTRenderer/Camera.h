#pragma once

#include "Ray.h"

class Camera
{
public:
	Vec3 lowerLeft;
	Vec3 horizonal;
	Vec3 vertical;
	Vec3 origin;

	Camera(int x, int y):
		lowerLeft(Vec3(-double(x) / double(y) * 5.0, -5, -10)),
		horizonal(Vec3(double(x) / double(y) * 10.0, 0, 0)),
		vertical(Vec3(0, 10, 0)),
		origin(Vec3(0, 0, 0))
	{}

	Ray getRay(double u, double v)
	{
		return Ray(origin, lowerLeft + u * horizonal + v * vertical - origin);
	}

};