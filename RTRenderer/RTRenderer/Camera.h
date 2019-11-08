#pragma once

#include "Ray.h"
#include <math.h>
#include <stdlib.h>

class Camera
{
public:
	Vec3 lowerLeft;
	Vec3 horizonal;
	Vec3 vertical;
	Vec3 origin;
	Vec3 u, v, w; // new base, u/v on cam plane, w point at lookDir

	double lensRadius;

	Camera(int x, int y, double vfov, Vec3 lookFrom, Vec3 lookAt, Vec3 vup, double aperture, double focus)
	{
		lensRadius = aperture / 2;
		double aspect = double(x) / double(y);
		double theta = vfov * 3.1415926535 / 180.0;
		double halfHeight = tan(theta / 2.00);
		double halfWidth = aspect * halfHeight;

		origin = lookFrom;
		w = unitVector(lookFrom - lookAt);
		u = unitVector(cross(vup, w));
		v = unitVector(cross(w, u));

		lowerLeft = origin
			- halfWidth * u * focus
			- halfHeight * v * focus
			- w * focus;
		horizonal = 2 * halfWidth * u * focus;
		vertical = 2 * halfHeight * v * focus;
	}

	Ray getRay(double s, double t)
	{
		Vec3 blur = lensRadius * randomVecInUnitDisk();
		Vec3 offset = u * blur.x() + v * blur.y();
		return Ray(origin + offset, lowerLeft + s * horizonal + t * vertical - origin - offset);
	}

};