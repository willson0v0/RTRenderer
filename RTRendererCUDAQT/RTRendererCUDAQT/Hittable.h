#pragma once

#include "misc.h"
#include "Ray.h"
#include "AABB.h"

class Material;

struct HitRecord
{
	float t;
	Vec3 point;
	Vec3 normal;
	Material* matPtr;
	float u;
	float v;
};

class Hittable
{
public:
	__device__ virtual bool hit(const Ray& r, float tMin, float tMax, HitRecord& rec, curandState* localRandState) const = 0;
	__device__ virtual bool boundingBox(AABB& box) const = 0;
};