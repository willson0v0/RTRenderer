#pragma once
#include "Hittable.h"
#include "misc.h"
#include <thrust/device_ptr.h>
#include <thrust/sort.h>

__device__ bool compare(const Hittable* a, const Hittable* b);
//__device__ void sortHittables(Hittable** list, int start, int stop);

class BVH : public Hittable
{
public:
	Hittable* left;
	Hittable* right;
	AABB box;

	BVH() {}
	BVH(Hittable** l, int n);

	__device__ virtual bool hit(const Ray& r, double tMin, double tMax, HitRecord& rec) const;
	__device__ virtual bool boundingBox(AABB& box) const;
};

__device__ bool BVH::hit(const Ray& r, double tMin, double tMax, HitRecord& rec) const
{
	if (box.hit(r, tMin, tMax))
	{
		HitRecord lRec, rRec;
		bool lHit = left->hit(r, tMin, tMax, lRec);
		bool rHit = right->hit(r, tMin, tMax, rRec);
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

__device__ BVH::BVH(Hittable** l, int n)
{
	printf("Building BVH tree for %p of length %d\r\nSorting", l, n);

	thrust::device_ptr<Hittable*> lt(l);
	thrust::sort(lt, lt + n, compare);
	//sortHittables(l, 0, n);
	

	for (int i = 0; i < n; i++)
	{
		AABB temp;
		l[i]->boundingBox(temp);
		printf("%.1lf\t", temp.near.e[0]);
	}

	printf("---DONE\r\n");

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
		left = new BVH(l, n / 2);
		right = new BVH(l + n / 2, n - n / 2);
	}

	AABB lBox, rBox;

	if (!left->boundingBox(lBox) || !right->boundingBox(rBox))
	{
		printf("bvh no bb");
	}

	box = surroundingBox(lBox, rBox);
}

__device__ bool BVH::boundingBox(AABB& b) const
{
	b = box;
	return true;
}
/*
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
		if (cBox.near.e[0] - pivitBox.far.e[0] > 0.0)
		{
			Hittable* ptr = list[largerEnd];
			list[largerEnd] = list[i];
			list[i] = ptr;
		}
		printf("%d  ", cBox.near.e[0]);
	}

	Hittable* ptr = list[largerEnd];
	list[largerEnd] = list[start];
	list[start] = ptr;

	sortHittables(list, start, largerEnd);
	sortHittables(list, largerEnd+1, stop);
}*/

__device__ bool compare(const Hittable* a, const Hittable* b)
{
	AABB lBox, rBox;
	
	if (!a->boundingBox(lBox) || !b->boundingBox(rBox))
	{
		printf("ohhhhhh\n");
	}

	if (lBox.near.e[0] - rBox.near.e[0] < 0.0)
	{
		return -1;
	}
	else
	{
		return 1;
	}
}