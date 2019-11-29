#pragma once

#include "misc.h"
#include "Ray.h"
#include "AABB.h"

class Material;

struct HitRecord
{
	double t;
	Vec3 point;
	Vec3 normal;
	Material* matPtr;
	double u;
	double v;
};

class Hittable
{
public:
	__device__ virtual bool hit(const Ray& r, double tMin, double tMax, HitRecord& rec) const = 0;
	__device__ virtual bool boundingBox(AABB& box) const = 0;
};