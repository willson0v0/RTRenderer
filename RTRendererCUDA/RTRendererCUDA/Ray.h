#pragma once

#include "misc.h"
#include "Vec3.h"

class Ray
{
public:
	Vec3 origin, direction;

	__device__ Ray() {}
	__device__ Ray(const Vec3& a, const Vec3& b):origin(a), direction(b) {}

	__device__ Vec3 pointAtParam(double t) const { return origin + t * direction; }
};