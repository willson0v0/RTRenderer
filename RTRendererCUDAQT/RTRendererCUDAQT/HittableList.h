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
	__device__ void append(Hittable** list, int n);
	__device__ void remove(int n);
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

__device__ void HittableList::append(Hittable** l, int n)
{
	Hittable** newList = new Hittable * [n + listSize];
	printMsg(LogLevel::debug, "Adding %d new object(s) to list", n);
	
	for (int i = 0; i < n + listSize; i++)
	{
		if (i < n)
		{
			newList[i] = list[i];
		}
		else
		{
			newList[i] = l[i - n];
		}
	}

	delete[] list;
	list = newList;
	listSize += n;
}

__device__ void HittableList::remove(int n)
{
	if (n > listSize)
	{
		printMsg(LogLevel::error, "Index out of bound.");
		return;
	}

	printMsg(LogLevel::debug, "Removing Index: %d", n);

	Hittable** newList = new Hittable * [listSize - 1];

	for (int i = 0; i < listSize; i++)
	{
		if (i < n)
		{
			newList[i] = list[i];
		}
		else if (i == n)
			continue;
		else
		{
			newList[i-1] = list[i];
		}
	}
	list = newList;
	listSize--;
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