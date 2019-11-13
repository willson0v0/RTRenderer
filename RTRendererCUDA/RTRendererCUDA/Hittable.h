#pragma once

#include "misc.h"
#include "Ray.h"

class Material;

struct HitRecord
{
	double t;
	Vec3 point;
	Vec3 norm;
	Material* matPtr;
	double u;
	double v;
};

class Hittable
{
public:
	__device__ virtual bool hit(const Ray& r, double tMin, double tMax, HitRecord& rec) const = 0;
};