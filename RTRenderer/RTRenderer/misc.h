#pragma once

#include <functional>
#include <random>
#include "Vec3.h"
#include "Ray.h"

inline double randD()
{
	static std::uniform_real_distribution<double> distribution(0.0, 1.0);
	static std::mt19937 generator;
	static std::function<double()> rand_generator =
		std::bind(distribution, generator);
	return rand_generator();
}

Vec3 randomVecInUnitSphere()
{
	Vec3 p;
	do
	{
		p = 2 * Vec3(randD(), randD(), randD());
	} while (p.squaredLength() >= 1);
	return p;
}

Vec3 reflect(const Vec3& v, const Vec3& norm)
{
	return v - norm * (2 * dot(v, norm));
}