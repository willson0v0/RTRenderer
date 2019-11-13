#pragma once

#include "Hittable.h"

class Sphere : public Hittable
{
public:
	Vec3 center;
	double radius;
	Material* matPtr;

	__device__ Sphere() {}
	__device__ Sphere(Vec3 center_, double radius_, Material *m):center(center_), radius(radius_), matPtr(m) {}

	__device__ virtual bool hit(const Ray& r, double tMin, double tMax, HitRecord& rec) const;

};


__device__ bool Sphere::hit(const Ray& r, double tMin, double tMax, HitRecord& rec) const
{
	Vec3 offset = r.origin - center;
	double a = dot(r.direction, r.direction);
	double b = dot(offset, r.direction);
	double c = dot(offset, offset) - radius * radius;
	double diss = b * b - a * c;

	if (diss > 0)
	{
		double t = (-b - sqrt(diss)) / a;
		if (t > tMin&& t < tMax)
		{
			rec.t = t;
			rec.point = r.pointAtParam(t);
			rec.norm = (rec.point - center) / radius;
			rec.matPtr = matPtr;
			getSphereUV((rec.point - center) / radius, rec.u, rec.v);
			return true;
		}
		t = (-b + sqrt(diss)) / a;
		if (t > tMin&& t < tMax)
		{
			rec.t = t;
			rec.point = r.pointAtParam(t);
			rec.norm = (rec.point - center) / radius;
			rec.matPtr = matPtr;
			getSphereUV((rec.point - center) / radius, rec.u, rec.v);
			return true;
		}
		return false;
	}
	else
	{
		return false;
	}
}

