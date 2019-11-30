#pragma once

#include "misc.h"
#include "Ray.h"


class Camera
{
public:
	Vec3 lowerLeft;
	Vec3 horizonal;
	Vec3 vertical;
	Vec3 origin;
	Vec3 u, v, w; // new base, u/v on cam plane, w point at lookDir

	double lensRadius;

	__device__ Camera(int x, int y, double vfov, Vec3 lookFrom, Vec3 lookAt, Vec3 vup, double aperture, double focus)
	{
		lensRadius = aperture / 2;
		double aspect = double(x) / double(y);
		double theta = vfov * PI / 180.0;
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
		printMsg(LogLevel::debug, "Camera Loaded. \n\t%d * %d, (%.2lf, %.2lf, %.2lf) -> (%.2lf, %.2lf, %.2lf), fov = %.2lf, aperture = %.2lf, focus distance = %.2lf",
			MAX_X,
			MAX_Y,
			lookFrom.e[0],
			lookFrom.e[1],
			lookFrom.e[2],
			lookAt.e[0],
			lookAt.e[1],
			lookAt.e[2],
			vfov,
			aperture,
			focus
		);
	}

	__device__ Ray getRay(double s, double t, curandState* localRandState)
	{
		Vec3 blur = lensRadius * randomVecInUnitDisk(localRandState);
		Vec3 offset = u * blur.e[0] + v * blur.e[1];
		return Ray(origin + offset, lowerLeft + s * horizonal + t * vertical - origin - offset);
	}

};