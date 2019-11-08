#pragma once

#include "Ray.h"

struct  hitRecord
{
	double t;
	Vec3 point;
	Vec3 norm;
};

class Hittable
{
public:
	virtual bool hit(const Ray& r, double tMin, double tMax, hitRecord& rec) const = 0;
};