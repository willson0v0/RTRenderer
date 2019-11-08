#pragma once

#include "Vec3.h"

class Ray
{
public:
	Ray() :A(Vec3()), B(Vec3()) {}
	Ray(const Vec3& a, const Vec3& b) :A(Vec3(a)), B(Vec3(b)) {}

	Vec3 origin() const { return A; }
	Vec3 direction() const { return B; }
	Vec3 pointAtParam(double t) const { return A + t*B; }

	Vec3 A;
	Vec3 B;
};