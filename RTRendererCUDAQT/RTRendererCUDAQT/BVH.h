#pragma once
#include "Hittable.h"
#include "misc.h"
#include <thrust/device_ptr.h>
#include <thrust/sort.h>

__device__ bool compareX(const Hittable* a, const Hittable* b);
__device__ bool compareY(const Hittable* a, const Hittable* b);
__device__ bool compareZ(const Hittable* a, const Hittable* b);
//__device__ void sortHittables(Hittable** list, int start, int stop);

class BVH : public Hittable
{
public:
	Hittable* left;
	Hittable* right;
	AABB box;

	__device__ BVH() {}
	__device__ BVH(Hittable** l, int n, curandState* localRandstate);

	__device__ virtual bool hit(const Ray& r, double tMin, double tMax, HitRecord& rec, curandState* localRandState) const;
	__device__ virtual bool boundingBox(AABB& box) const;
};

__device__ bool BVH::hit(const Ray& r, double tMin, double tMax, HitRecord& rec, curandState* localRandState) const
{
	if (box.hit(r, tMin, tMax))
	{
		HitRecord lRec, rRec;
		bool lHit = left->hit(r, tMin, tMax, lRec, localRandState);
		bool rHit = right->hit(r, tMin, tMax, rRec, localRandState);
		if (lHit && rHit)
		{
			rec = (lRec.t < rRec.t ? lRec : rRec);
			return true;
		}
		else if (lHit)
		{
			rec = lRec;
			return true;
		}
		else if (rHit)
		{
			rec = rRec;
			return true;
		}
		else
		{
			return false;
		}
	}
	else
	{
		return false;
	}
}

__device__ BVH::BVH(Hittable** l, int n, curandState* localRandState)
{
	curandState rs = *localRandState;

	thrust::device_ptr<Hittable*> lt(l);

	int div = 3 * curand_uniform(localRandState);
	switch (div)
	{
	case 0:
		thrust::sort(lt, lt + n, compareX);
		break;
	case 1:
		thrust::sort(lt, lt + n, compareY);
		break;
	case 2:
	default:
		thrust::sort(lt, lt + n, compareZ);
		break;
	}

	if (n == 1)
	{
		left = right = l[0];
	}
	else if (n == 2)
	{
		left = l[0];
		right = l[1];
	}
	else
	{
		left = new BVH(l, n / 2, localRandState);
		right = new BVH(l + n / 2, n - n / 2, localRandState);
	}

	AABB lBox, rBox;

	if (!left->boundingBox(lBox) || !right->boundingBox(rBox))
	{
		printf("warning: bvh no bb");
	}

	box = surroundingBox(lBox, rBox);
	*localRandState = rs;
}

__device__ bool BVH::boundingBox(AABB& b) const
{
	b = box;
	return true;
}

//vvvvv deprecated: use thrust lib instead. vvvvv
__device__ void sortHittables(Hittable** list, int start, int stop)
{
	if (stop - start <= 1) return;
	printf("%p, %d, %d\r\n", list, start, stop);
	int largerEnd = start;
	for (int i = start; i < stop; i++)
	{
		AABB cBox, pivitBox;
		if (!list[i]->boundingBox(cBox) || !list[start]->boundingBox(pivitBox))
		{
			printf("No bb @ sort");
		}
		if (cBox.nearVec.e[0] - pivitBox.farVec.e[0] > 0.0)
		{
			Hittable* ptr = list[largerEnd];
			list[largerEnd] = list[i];
			list[i] = ptr;
		}
		printf("%d  ", cBox.nearVec.e[0]);
	}

	Hittable* ptr = list[largerEnd];
	list[largerEnd] = list[start];
	list[start] = ptr;

	sortHittables(list, start, largerEnd);
	sortHittables(list, largerEnd+1, stop);
}

__device__ bool compareX(const Hittable* a, const Hittable* b)
{
	AABB lBox, rBox;
	if (!a->boundingBox(lBox) || !b->boundingBox(rBox))
	{
		printf("Warning: No Bounding box in constructor\n");
	}
	return (lBox.nearVec.e[0] < rBox.nearVec.e[0]);
}

__device__ bool compareY(const Hittable* a, const Hittable* b)
{
	AABB lBox, rBox;
	if (!a->boundingBox(lBox) || !b->boundingBox(rBox))
	{
		printf("Warning: No Bounding box in constructor\n");
	}
	return (lBox.nearVec.e[1] < rBox.nearVec.e[1]);
}

__device__ bool compareZ(const Hittable* a, const Hittable* b)
{
	AABB lBox, rBox;
	if (!a->boundingBox(lBox) || !b->boundingBox(rBox))
	{
		printf("Warning: No Bounding box in constructor\n");
	}
	return (lBox.nearVec.e[2] < rBox.nearVec.e[2]);
}