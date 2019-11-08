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

	Metal(double r, double g, double b) : albedo(Vec3(r, g, b)) {}
	Metal(const Vec3& a) : albedo(a) {}
	virtual bool scatter(const Ray& rIn, const hitRecord& rec, Vec3& attenuation, Ray& scattered) const
	{
		Vec3 reflected = reflect(unitVector(rIn.direction()), rec.norm);
		scattered = Ray(rec.point, reflected);
		attenuation = albedo;
		return (dot(scattered.direction(), rec.norm) > 0);
	}
};