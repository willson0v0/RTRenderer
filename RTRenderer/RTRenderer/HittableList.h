#pragma once
#include "Hittable.h"

class HittableList : public Hittable
{
public:
	Hittable** list;	// Could shift to link list
	int listSize;

	HittableList():list(nullptr), listSize(0) {}
	HittableList(Hittable** l, int n):list(l), listSize(n) {}
	virtual bool hit(const Ray& r, double tMin, double tMax, hitRecord& rec) const;
};

bool HittableList::hit(const Ray& r, double tMin, double tMax, hitRecord& rec) const
{
	hitRecord tRec;
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