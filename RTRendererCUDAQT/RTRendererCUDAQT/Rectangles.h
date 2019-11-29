#pragma once
#include "Hittable.h"
#include "Material.h"

class FlipNorm :public Hittable
{
public:
	Hittable* content;
	__device__ FlipNorm(Hittable* p) :content(p) {}

	__device__ virtual bool hit(const Ray& r, double tMin, double tMax, HitRecord& rec, curandState* localRandState) const
	{
		if (content->hit(r, tMin, tMax, rec, localRandState))
		{
			rec.normal = -rec.normal;
			return true;
		}
		else
		{
			return false;
		}
	}

	__device__ bool virtual boundingBox(AABB& box) const;
};

__device__ bool FlipNorm::boundingBox(AABB& box) const
{
	return content->boundingBox(box);
}

class RectXY :public Hittable
{
public:
	Material* matPtr;
	double x0, x1, y0, y1, z, xL, yL;

	__device__ RectXY() {}
	__device__ RectXY(double x0_, double x1_, double y0_, double y1_, double z_, Material* matPtr_)
		: x0(x0_), x1(x1_), y0(y0_), y1(y1_), z(z_), xL(x1 - x0), yL(y1 - y0), matPtr(matPtr_) {}

	__device__ virtual bool hit(const Ray& r, double tMin, double tMax, HitRecord& rec, curandState* localRandState) const
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
		rec.normal = Vec3(0, 0, 1);
		return true;
	}
	__device__ bool virtual boundingBox(AABB& box) const
	{
		box = AABB(Vec3(x0, y0, z - 0.0001), Vec3(x1, y1, z + 0.0001));
		return true;
	}
};


class RectXZ :public Hittable
{
public:
	Material* matPtr;
	double x0, x1, z0, z1, y, xL, zL;

	__device__ RectXZ() {}
	__device__ RectXZ(double x0_, double x1_, double z0_, double z1_, double y_, Material* matPtr_)
		: x0(x0_), x1(x1_), z0(z0_), z1(z1_), y(y_), xL(x1 - x0), zL(z1 - z0), matPtr(matPtr_) {}

	__device__ virtual bool hit(const Ray& r, double tMin, double tMax, HitRecord& rec, curandState* localRandState) const
	{
		//				r.origin.y			r.dir.y
		double t = (y - r.origin.e[1]) / r.direction.e[1];

		double x = r.origin.e[0] + t * r.direction.e[0];
		double z = r.origin.e[2] + t * r.direction.e[2];

		if (t<tMin || t>tMax || x<x0 || x>x1 || z<z0 || z>z1)
			return false;

		rec.u = (x - x0) / xL;
		rec.v = (z - z0) / zL;
		rec.t = t;
		rec.matPtr = matPtr;
		rec.point = r.pointAtParam(t);
		rec.normal = Vec3(0, 1, 0);
		return true;
	}
	__device__ bool virtual boundingBox(AABB& box) const
	{
		box = AABB(Vec3(x0, y - 0.0001, z0), Vec3(x1, y + 0.0001, z1));
		return true;
	}
};

class RectYZ :public Hittable
{
public:
	Material* matPtr;
	double y0, y1, z0, z1, x, yL, zL;

	__device__ RectYZ() {}
	__device__ RectYZ(double y0_, double y1_, double z0_, double z1_, double x_, Material* matPtr_)
		: y0(y0_), y1(y1_), z0(z0_), z1(z1_), x(x_), yL(y1 - y0), zL(z1 - z0), matPtr(matPtr_) {}

	__device__ virtual bool hit(const Ray& r, double tMin, double tMax, HitRecord& rec, curandState* localRandState) const
	{
		//				r.origin.x			r.dir.x
		double t = (x - r.origin.e[0]) / r.direction.e[0];

		double y = r.origin.e[1] + t * r.direction.e[1];
		double z = r.origin.e[2] + t * r.direction.e[2];

		if (t<tMin || t>tMax || y<y0 || y>y1 || z<z0 || z>z1)
			return false;

		rec.u = (y - y0) / yL;
		rec.v = (z - z0) / zL;
		rec.t = t;
		rec.matPtr = matPtr;
		rec.point = r.pointAtParam(t);
		rec.normal = Vec3(1, 0, 0);
		return true;
	}
	__device__ bool virtual boundingBox(AABB& box) const
	{
		box = AABB(Vec3(x - 0.0001, y0, z0), Vec3(x + 0.0001, y1, z1));
		return true;
	}
};