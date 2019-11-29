#pragma once

#include "Hittable.h"
#include "Material.h"

class ConstantMedium : public Hittable
{
public:
	Hittable* boundary;
	double density;
	Material* phaseFunc;

	__device__ ConstantMedium(Hittable* b, double d, Texture* a) : boundary(b), density(d), phaseFunc(new Isotropic(a))	{}

	__device__ virtual bool hit(const Ray& r, double tMin, double tMax, HitRecord& rec, curandState* localRandState) const;
	__device__ virtual bool boundingBox(AABB& box) const
	{
		return boundary->boundingBox(box);
	}
};

__device__ bool ConstantMedium::hit(const Ray& r, double tMin, double tMax, HitRecord& rec, curandState* localRandState) const
{
	HitRecord rec1, rec2;
	if (boundary->hit(r, -DBL_MAX, DBL_MAX, rec1, localRandState))
	{
		if (boundary->hit(r, rec1.t + 0.000001, DBL_MAX, rec2, localRandState))
		{
			rec1.t = rec1.t < tMin ? tMin : rec1.t;
			rec2.t = rec2.t > tMax ? tMax : rec2.t;
			if (rec1.t >= rec2.t) return false;
			rec1.t = rec1.t < 0 ? 0 : rec1.t;

			double distance = (rec2.t - rec1.t) * r.direction.length();
			double hitAt = - (1/density) * log(curand_uniform(localRandState));


			if (hitAt < distance)
			{
				rec.t = rec1.t + hitAt / r.direction.length();
				rec.point = r.pointAtParam(rec.t);
				rec.normal = Vec3(1, 0, 0); // what so ever.
				rec.matPtr = phaseFunc;
				return true;
			}
		}
	}
	return false;
}

