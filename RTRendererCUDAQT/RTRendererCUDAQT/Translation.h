#pragma once

#include "Hittable.h"

class Translate : public Hittable
{
public:
	Hittable* origin;
	Vec3 offset;

	__device__ Translate(Hittable* p, const Vec3& ofst) : origin(p), offset(ofst) {}

	__device__ virtual bool hit(const Ray& r, double tMin, double tMax, HitRecord& rec) const
	{
		Ray moved(r.origin - offset, r.direction);
		if (origin->hit(moved, tMin, tMax, rec))
		{
			rec.point += offset;
			return true;
		}
		else
		{
			return false;
		}
	}
	
	__device__ virtual bool boundingBox(AABB& box) const
	{
		if (origin->boundingBox(box))
		{
			box.far += offset;
			box.near += offset;
			return true;
		}
		else return false;
	}
};

class RotateY : public Hittable
{
public:
	Hittable* origin;
	double sinTheta;
	double cosTheta;
	bool hasBB;
	AABB bBox;

	__device__ RotateY(Hittable* p, double angle):origin(p)
	{
		double radians = (PI / 180.0) * angle;
		sinTheta = sin(radians);
		cosTheta = cos(radians);

		hasBB = origin->boundingBox(bBox);

		double x[4];
		double z[4];

		x[0] = cosTheta * bBox.far.e[0] + sinTheta * bBox.far.e[2];
		x[1] = cosTheta * bBox.far.e[0] + sinTheta * bBox.near.e[2];
		x[2] = cosTheta * bBox.near.e[0] + sinTheta * bBox.far.e[2];
		x[3] = cosTheta * bBox.near.e[0] + sinTheta * bBox.near.e[2];

		z[0] = -sinTheta * bBox.far.e[0] + cosTheta * bBox.far.e[2];
		z[1] = -sinTheta * bBox.far.e[0] + cosTheta * bBox.near.e[2];
		z[2] = -sinTheta * bBox.near.e[0] + cosTheta * bBox.far.e[2];
		z[3] = -sinTheta * bBox.near.e[0] + cosTheta * bBox.near.e[2];

		double xMin = DBL_MAX;
		double xMax = DBL_MIN;
		double yMin = DBL_MAX;
		double yMax = DBL_MIN;
		double zMin = DBL_MAX;
		double zMax = DBL_MIN;

		for (int i = 0; i < 4; i++)
		{
			xMin = xMin < x[i] ? xMin : x[i];
			xMax = xMax > x[i] ? xMax : x[i];
			zMin = zMin < z[i] ? zMin : z[i];
			zMax = zMax > z[i] ? zMax : z[i];
		}

		if (bBox.far.e[1] > bBox.near.e[1])
		{
			yMin = bBox.near.e[1];
			yMax = bBox.far.e[1];
		}
		else
		{
			yMin = bBox.far.e[1];
			yMax = bBox.near.e[1];
		}

		bBox = AABB(Vec3(xMin, yMin, zMin), Vec3(xMax, yMax, zMax));
	}

	__device__ virtual bool hit(const Ray& r, double tMin, double tMax, HitRecord& rec) const
	{
		Ray rotated(
			Vec3(
				cosTheta * r.origin.e[0] - sinTheta * r.origin.e[2],
				r.origin.e[1],
				sinTheta * r.origin.e[0] + cosTheta * r.origin.e[2]
			), Vec3(
				cosTheta * r.direction.e[0] - sinTheta * r.direction.e[2],
				r.direction.e[1],
				sinTheta * r.direction.e[0] + cosTheta * r.direction.e[2]
			)
		);

		if (origin->hit(rotated, tMin, tMax, rec))
		{
			Vec3 p(
				cosTheta * rec.point.e[0] + sinTheta * rec.point.e[2],
				rec.point.e[1],
				-sinTheta * rec.point.e[0] + cosTheta * rec.point.e[2]
			);
			Vec3 norm(
				cosTheta * rec.normal.e[0] + sinTheta * rec.normal.e[2],
				rec.normal.e[1],
				-sinTheta * rec.normal.e[0] + cosTheta * rec.normal.e[2]
			);

			rec.point = p;
			rec.normal = norm;
			
			return true;
		}
		else return false;
	}

	__device__ virtual bool boundingBox(AABB& box) const
	{
		box = bBox;
		return hasBB;
	}
};

