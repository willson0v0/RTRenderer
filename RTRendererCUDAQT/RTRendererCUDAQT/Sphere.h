#pragma once

#include "Hittable.h"

class Sphere : public Hittable
{
public:
	Vec3 center;
	float radius;
	Material* matPtr;

	__device__ Sphere() {}
	__device__ Sphere(Vec3 center_, float radius_, Material *m):center(center_), radius(radius_), matPtr(m) {}

	__device__ virtual bool hit(const Ray& r, float tMin, float tMax, HitRecord& rec, curandState* localRandState) const;
	__device__ virtual bool boundingBox(AABB& box) const;

};

__device__ bool Sphere::hit(const Ray& r, float tMin, float tMax, HitRecord& rec, curandState* localRandState) const
{
	Vec3 offset = r.origin - center;
	float a = dot(r.direction, r.direction);
	float b = dot(offset, r.direction);
	float c = dot(offset, offset) - radius * radius;
	float diss = b * b - a * c;

	if (diss > 0)
	{
		float t = (-b - sqrt(diss)) / a;
		if (t > tMin&& t < tMax)
		{
			rec.t = t;
			rec.point = r.pointAtParam(t);
			rec.normal = (rec.point - center) / radius;
			rec.matPtr = matPtr;
			getSphereUV((rec.point - center) / radius, rec.u, rec.v);
			return true;
		}
		t = (-b + sqrt(diss)) / a;
		if (t > tMin&& t < tMax)
		{
			rec.t = t;
			rec.point = r.pointAtParam(t);
			rec.normal = (rec.point - center) / radius;
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

__device__ bool Sphere::boundingBox(AABB& box) const 
{
	box = AABB(center - Vec3(radius, radius, radius), center + Vec3(radius, radius, radius));
	return true;
}