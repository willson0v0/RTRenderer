#pragma once
#include "misc.h"
#include "Ray.h"

class AABB
{
public:
	Vec3 nearVec;
	Vec3 farVec;

	__device__ AABB() {}
	__device__ AABB(const Vec3& near_, const Vec3& far_): nearVec(near_), farVec(far_) {}
	__device__ bool hit(const Ray& r, double tMin, double tMax) const
	{
		for (int i = 0; i < 3; i++)
		{
			double invD = 1.0 / r.direction.e[i];
			double t0 = (nearVec.e[i] - r.origin.e[i]) * invD;
			double t1 = (farVec.e[i] - r.origin.e[i]) * invD;
			if (invD < 0)
			{
				double t = t0;
				t0 = t1;
				t1 = t;
			}
			tMin = t0 > tMin ? t0 : tMin;
			tMax = t1 < tMax ? t1 : tMax;
			if (tMax <= tMin) return false;
		}
		return true;
	}
};

__device__ AABB surroundingBox(AABB box0, AABB box1)
{
	Vec3 nearVec(
		ffmin(box0.nearVec.e[0], box1.nearVec.e[0]),
		ffmin(box0.nearVec.e[1], box1.nearVec.e[1]),
		ffmin(box0.nearVec.e[2], box1.nearVec.e[2])
	);

	Vec3 farVec(
		ffmax(box0.farVec.e[0], box1.farVec.e[0]),
		ffmax(box0.farVec.e[1], box1.farVec.e[1]),
		ffmax(box0.farVec.e[2], box1.farVec.e[2])
	);

	return AABB(nearVec, farVec);
}