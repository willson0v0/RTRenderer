#pragma once

#include "Hittable.h"
#include "Ray.h"
#include "misc.h"

class Material
{
public:
	virtual bool scatter(const Ray& rIn, const hitRecord& rec, Vec3& attenuation, Ray& scattered) const = 0;
};

class Lambertian : public Material
{
public:
	Vec3 albedo;

	Lambertian(double r, double g, double b) : albedo(Vec3(r, g, b)) {}
	Lambertian(const Vec3& a) : albedo(a) {}
	virtual bool scatter(const Ray& rIn, const hitRecord& rec, Vec3& attenuation, Ray& scattered) const
	{
		Vec3 tgt = rec.norm + randomVecInUnitSphere();
		scattered = Ray(rec.point, tgt);
		attenuation = albedo;
		return true;
	}
};

class Metal : public Material
{
public:
	Vec3 albedo;
	double fuzz;

	Metal(double r, double g, double b, double f) : albedo(Vec3(r, g, b)), fuzz(f < 1 ? f : 1) {}
	Metal(const Vec3& a, double f) : albedo(a), fuzz(f) {}
	virtual bool scatter(const Ray& rIn, const hitRecord& rec, Vec3& attenuation, Ray& scattered) const
	{
		Vec3 reflected = reflect(unitVector(rIn.direction()), rec.norm);
		scattered = Ray(rec.point, reflected + fuzz*randomVecInUnitSphere());
		attenuation = albedo;
		return (dot(scattered.direction(), rec.norm) > 0);
	}
};

class dielectric : public Material
{
public:
	double refIndex;
	double fuzz;
	Vec3 albedo;

	dielectric(Vec3 al, double ri, double f) : refIndex(ri), albedo(al), fuzz(f>1?1:f) {}
	dielectric(double r, double g, double b, double ri, double f) : refIndex(ri), albedo(Vec3(r, g, b)), fuzz(f>1?1:f) {}

	virtual bool scatter(const Ray& rIn, const hitRecord& rec, Vec3& attenuation, Ray& scattered) const
	{
		Vec3 oNorm;
		Vec3 reflected = reflect(rIn.direction(), rec.norm);
		double rri;
		attenuation = albedo;
		Vec3 refracted;

		double reflectProb;
		double cos;

		if (dot(rIn.direction(), rec.norm) > 0) // Light went into glass
		{
			oNorm = -rec.norm;
			rri = refIndex;
			cos = refIndex * dot(rIn.direction(), rec.norm) / rIn.direction().length();
		}
		else
		{
			oNorm = rec.norm;
			rri = 1.0 / refIndex;
			cos = -dot(rIn.direction(), rec.norm) / rIn.direction().length();
		}

		if (refract(rIn.direction(), oNorm, rri, refracted))
		{
			reflectProb = schlick(cos, refIndex);
		}
		else
		{
			reflectProb = 1.0;
		}

		if(randD() < reflectProb)
		{
			scattered = Ray(rec.point, reflected + fuzz * randomVecInUnitSphere());
		}
		else
		{
			scattered = Ray(rec.point, refracted + fuzz * randomVecInUnitSphere());
		}
		return true;
	}
};