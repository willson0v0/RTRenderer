#pragma once
#include "Vec3.h"
#include "Ray.h"
#include "Hittable.h"
#include "misc.h"

class Sphere: public Hittable
{
public:
	Vec3 center;
	double radius;
	Material* matPtr;

	Sphere(double x, double y, double z, double r, Material* m) :center(x, y, z), radius(r), matPtr(m) {}
	Sphere(Vec3 c, double r, Material* m) :center(c), radius(r), matPtr(m) {}

	virtual bool hit(const Ray& r, double tMin, double tMax, hitRecord& rec) const;
};

bool Sphere::hit(const Ray& r, double tMin, double tMax, hitRecord& rec) const
{
	Vec3 offset = r.A - center;
	double a = dot(r.B, r.B);
	double b = dot(offset, r.B);
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
			rec.matPtr = matPtr;
			getSphereUV((rec.point - center) / radius, rec.u, rec.v);
			return true;
		}
		t = (-b + sqrt(diss)) / a;
		if (t > tMin&& t < tMax)
		{
			rec.t = t;
			rec.point = r.pointAtParam(t);
			rec.norm = (rec.point - center) / radius;
			rec.matPtr = matPtr;
			getSphereUV((rec.point - center) / radius, rec.u, rec.v);
			return true;
		}
		return false;
	}
	else
	{
		return false;
	}
}