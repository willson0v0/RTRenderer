#pragma once

#include "Hittable.h"

class HittableList : public Hittable
{
public:
	Hittable** list;	// Could shift to link list
	int listSize;

	__device__ HittableList() :list(nullptr), listSize(0) {}
	__device__ HittableList(Hittable** l, int n) :list(l), listSize(n) {}
	__device__ virtual bool hit(const Ray& r, double tMin, double tMax, HitRecord& rec) const;
};

__device__ bool HittableList::hit(const Ray& r, double tMin, double tMax, HitRecord& rec) const
{
	HitRecord tRec;
	bool hitAny = false;
	double closest = tMax;
	for (int i = 0; i < listSize; i++)
	{
		if (list[i]->hit(r, tMin, closest, tRec))
		{
			hitAny = true;
			closest = tRec.t;
			rec = tRec;
		}
	}
	return hitAny;
}