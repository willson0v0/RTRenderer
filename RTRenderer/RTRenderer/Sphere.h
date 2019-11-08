#pragma once
#include "Vec3.h"
#include "Ray.h"
#include "Hittable.h"

class Sphere: public Hittable
{
public:
	Vec3 center;
	double radius;

	Sphere(double x, double y, double z, double r) :center(x, y, z), radius(r) {}
	Sphere(Vec3 c, double r) :center(c), radius(r) {}

	virtual bool hit(const Ray& r, double tMin, double tMax, hitRecord& rec) const;
};

bool Sphere::hit(const Ray& r, double tMin, double tMax, hitRecord& rec) const
{
	Vec3 offset = r.origin() - center;
	double a = dot(r.direction(), r.direction());
	double b = dot(offset, r.direction());
	double c = dot(offset, offset) - radius * radius;
	double diss = b * b - a * c;
	if (diss > 0)
	{
		double t = (-b - sqrt(diss)) / a;
		if (t > tMin && t < tMax)
		{
			rec.t = t;
			rec.point = r.pointAtParam(t);
			rec.norm = (rec.point - center) / radius;
			return true;
		}
		t = (-b + sqrt(diss)) / a;
		if (t > tMin&& t < tMax)
		{
			rec.t = t;
			rec.point = r.pointAtParam(t);
			rec.norm = (rec.point - center) / radius;
			return true;
		}
		return false;
	}
	else
	{
		return false;
	}
}