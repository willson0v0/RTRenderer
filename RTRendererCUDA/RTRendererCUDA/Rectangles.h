#pragma once
#include "Hittable.h"
#include "Material.h"

class FlipNorm :public Hittable
{
public:
	Hittable* content;
	__device__ FlipNorm(Hittable* p) :content(p) {}

	__device__ virtual bool hit(const Ray& r, double tMin, double tMax, HitRecord& rec) const
	{
		if (content->hit(r, tMin, tMax, rec))
		{
			rec.norm = -rec.norm;
			return true;
		}
		else
		{
			return false;
		}
	}
};

class RectXY :public Hittable
{
public:
	Material* matPtr;
	double x0, x1, y0, y1, z, xL, yL;

	__device__ RectXY() {}
	__device__ RectXY(double x0_, double x1_, double y0_, double y1_, double z_, Material* matPtr_)
		: x0(x0_), x1(x1_), y0(y0_), y1(y1_), z(z_), xL(x1 - x0), yL(y1 - y0), matPtr(matPtr_) {}

	__device__ virtual bool hit(const Ray& r, double tMin, double tMax, HitRecord& rec) const;
};

__device__ bool RectXY::hit(const Ray& r, double tMin, double tMax, HitRecord& rec) const
{
	//				r.origin.z  r.dir.z
	double t = (z - r.origin.e[2]) / r.direction.e[2];

	double x = r.origin.e[0] + t * r.direction.e[0];
	double y = r.origin.e[1] + t * r.direction.e[1];

	if (t<tMin || t>tMax || x<x0 || x>x1 || y<y0 || y>y1)
		return false;

	rec.u = (x - x0) / xL;
	rec.v = (y - y0) / yL;
	rec.t = t;
	rec.matPtr = matPtr;
	rec.point = r.pointAtParam(t);
	rec.norm = Vec3(0, 0, 1);
	return true;
}