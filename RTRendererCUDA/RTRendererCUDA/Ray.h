#pragma once

#include "misc.h"
#include "Vec3.h"

class Ray
{
public:
	Vec3 A, B;

	__device__ Ray() {}
	__device__ Ray(const Vec3& a, const Vec3& b):A(a), B(b) {}

	__device__ Vec3 origin() const { return A; }
	__device__ Vec3 direction() const { return B; }
	__device__ Vec3 pointAtParam(double t) const { return A + t * B; }
};