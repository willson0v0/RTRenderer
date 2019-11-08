#pragma once

#include <functional>
#include <random>
#include "Vec3.h"
#include "Ray.h"
#include "Material.h"

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

Vec3 randomVecInUnitDisk()
{
	Vec3 p;
	do
	{
		p = 2 * Vec3(randD(), randD(), 0) - Vec3(1,1,0);
	} while (p.squaredLength() >= 1);
	return p;
}

Vec3 reflect(const Vec3& v, const Vec3& norm)
{
	return v - norm * (2 * dot(v, norm));
}

bool refract(const Vec3& v, const Vec3& n, double rri, Vec3& refracted) // rri: relative refractive index.
{
	Vec3 uv = unitVector(v);
	double dt = dot(uv, n);  //in angle
	double dis = 1 - rri * rri * (1 - dt * dt); // is total internal reflection?
	if (dis > 0)
	{
		refracted = rri * (uv - n * dt) - n * sqrt(dis);
		return true;
	}
	return false;
}

double schlick(double cosine, double refIndex) {
	double r0 = (1 - refIndex) / (1 + refIndex);
	r0 = r0 * r0;
	return r0 + (1 - r0) * pow((1 - cosine), 5);
}
