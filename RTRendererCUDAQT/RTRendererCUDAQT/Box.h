#pragma once

#include "Hittable.h"
#include "HittableList.h"
#include "Rectangles.h"

class Box : public Hittable
{
public:
	Vec3 pMin, pMax;
	Hittable* surfaces;

	__device__ Box() {}
	__device__ Box(const Vec3& p0, const Vec3& p1, Material* ptr);

	__device__ virtual bool hit(const Ray& r, double tMin, double tMax, HitRecord& rec) const;
	__device__ virtual bool boundingBox(AABB& box) const;
};

__device__ Box::Box(const Vec3& p0, const Vec3& p1, Material* ptr)
{
	pMin = p0;
	pMax = p1;
	Hittable** list = new Hittable* [6];
	list[0] = new RectXY(pMin.e[0], pMax.e[0], pMin.e[1], pMax.e[1], pMax.e[2], ptr);
	list[1] = new RectXY(pMin.e[0], pMax.e[0], pMin.e[1], pMax.e[1], pMin.e[2], ptr);
	list[2] = new RectXZ(pMin.e[0], pMax.e[0], pMin.e[2], pMax.e[2], pMax.e[1], ptr);
	list[3] = new RectXZ(pMin.e[0], pMax.e[0], pMin.e[2], pMax.e[2], pMin.e[1], ptr);
	list[4] = new RectYZ(pMin.e[1], pMax.e[1], pMin.e[2], pMax.e[2], pMax.e[0], ptr);
	list[5] = new RectYZ(pMin.e[1], pMax.e[1], pMin.e[2], pMax.e[2], pMin.e[0], ptr);

	list[1] = new FlipNorm(list[1]);
	list[3] = new FlipNorm(list[3]);
	list[5] = new FlipNorm(list[5]);

	surfaces = new HittableList(list, 6);
}

__device__ bool Box::hit(const Ray& r, double tMin, double tMax, HitRecord& rec) const
{
	return surfaces->hit(r, tMin, tMax, rec);
}

__device__ bool Box::boundingBox(AABB& box) const
{
	box = AABB(pMin, pMax);
	return true;
}
