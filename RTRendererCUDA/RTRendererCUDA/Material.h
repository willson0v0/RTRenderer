#pragma once

#include "Hittable.h"
#include "Texture.h"

class Material
{
public:
	__device__ virtual bool scatter(const Ray& rIn, const HitRecord& rec, Vec3& attenuation, Ray& scattered, curandState* localRandState) const = 0;
	//virtual Vec3 emitted(double u, double v, const Vec3& p) const { return Vec3(0, 0, 0); }
};

class Lambertian : public Material
{
public:
	Texture* albedo;

	__device__ Lambertian(Texture* a) : albedo(a) {}
	__device__ virtual bool scatter(const Ray& rIn, const HitRecord& rec, Vec3& attenuation, Ray& scattered, curandState* localRandState) const
	{
		Vec3 tgt = rec.norm + randomVecInUnitSphere(localRandState);
		scattered = Ray(rec.point, tgt);
		attenuation = albedo->value(rec.u, rec.v, rec.point);
		return true;
	}
};

class Metal : public Material
{
public:
	Vec3 albedo;
	double fuzz;

	__device__ Metal(double r, double g, double b, double f) : albedo(Vec3(r, g, b)), fuzz(f < 1 ? f : 1) {}
	__device__ Metal(const Vec3& a, double f) : albedo(a), fuzz(f) {}
	__device__ virtual bool scatter(const Ray& rIn, const HitRecord& rec, Vec3& attenuation, Ray& scattered, curandState* localRandState) const
	{
		Vec3 reflected = reflect(unitVector(rIn.direction()), rec.norm);
		scattered = Ray(rec.point, reflected + fuzz * randomVecInUnitSphere(localRandState));
		attenuation = albedo;
		return (dot(scattered.direction(), rec.norm) > 0);
	}
};

class Dielectric : public Material
{
public:
	double refIndex;
	double fuzz;
	Vec3 albedo;

	__device__ Dielectric(Vec3 al, double ri, double f) : refIndex(ri), albedo(al), fuzz(f > 1 ? 1 : f) {}

	__device__ virtual bool scatter(const Ray& rIn, const HitRecord& rec, Vec3& attenuation, Ray& scattered, curandState* localRandState) const
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

		if (curand_uniform(localRandState) < reflectProb)
		{
			scattered = Ray(rec.point, reflected + fuzz * randomVecInUnitSphere(localRandState));
		}
		else
		{
			scattered = Ray(rec.point, refracted + fuzz * randomVecInUnitSphere(localRandState));
		}
		return true;
	}
};