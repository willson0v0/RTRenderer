#pragma once

#include "Hittable.h"

class HittableList : public Hittable
{
public:
	Hittable** list;	// Could shift to link list
	int listSize;

	__device__ HittableList() :list(nullptr), listSize(0) {}
	__device__ HittableList(Hittable** l, int n) :list(l), listSize(n) {}
	__device__ virtual bool hit(const Ray& r, float tMin, float tMax, HitRecord& rec, curandState* localRandState) const;
	__device__ bool virtual boundingBox(AABB& box) const;
};

__device__ bool HittableList::hit(const Ray& r, float tMin, float tMax, HitRecord& rec, curandState* localRandState) const
{
	HitRecord tRec;
	bool hitAny = false;
	float closest = tMax;
	for (int i = 0; i < listSize; i++)
	{
		if (list[i]->hit(r, tMin, closest, tRec, localRandState))
		{
			hitAny = true;
			closest = tRec.t;
			rec = tRec;
		}
	}
	return hitAny;
}

__device__ bool HittableList::boundingBox(AABB& box) const
{
	if (listSize < 1) return false;
	AABB temp;
	if (!list[0]->boundingBox(temp))
	{
		return false;
	}
	else
	{
		box = temp;
	}
	for (int i = 1; i < listSize; i++)
	{
		if (list[i]->boundingBox(temp))
		{
			box = surroundingBox(temp, box);
		}
		else return false;
	}
	return true;
}