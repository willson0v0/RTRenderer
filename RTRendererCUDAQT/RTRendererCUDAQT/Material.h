#pragma once

#include "Hittable.h"
#include "Texture.h"

class Material
{
public:
	__device__ virtual bool scatter(const Ray& rIn, const HitRecord& rec, Vec3& attenuation, Ray& scattered, curandState* localRandState) const = 0;
	//virtual Vec3 emitted(double u, double v, const Vec3& p) const { return Vec3(0, 0, 0); }

	__device__ virtual Vec3 emitted(double u, double v, const Vec3& point) const
	{
		return Vec3(0, 0, 0);
	}
};

class Lambertian : public Material
{
public:
	Texture* albedo;

	__device__ Lambertian(Texture* a) : albedo(a) {}
	__device__ virtual bool scatter(const Ray& rIn, const HitRecord& rec, Vec3& attenuation, Ray& scattered, curandState* localRandState) const
	{
		Vec3 tgt = rec.normal + randomVecInUnitSphere(localRandState);
		scattered = Ray(rec.point, tgt);
		attenuation = albedo->value(rec.u, rec.v, rec.point);
		return true;
	}

	__device__ virtual Vec3 emitted(double u, double v, const Vec3& point) const
	{
		return Vec3(0, 0, 0);
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
		Vec3 reflected = reflect(unitVector(rIn.direction), rec.normal);
		scattered = Ray(rec.point, reflected + fuzz * randomVecInUnitSphere(localRandState));
		attenuation = albedo;
		return (dot(scattered.direction, rec.normal) > 0);
	}

	__device__ virtual Vec3 emitted(double u, double v, const Vec3& point) const
	{
		return Vec3(0, 0, 0);
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
		Vec3 reflected = reflect(rIn.direction, rec.normal);
		double rri;
		attenuation = albedo;
		Vec3 refracted;

		double reflectProb;
		double cos;

		if (dot(rIn.direction, rec.normal) > 0) // Light went into glass
		{
			oNorm = -rec.normal;
			rri = refIndex;
			cos = refIndex * dot(rIn.direction, rec.normal) / rIn.direction.length();
		}
		else
		{
			oNorm = rec.normal;
			rri = 1.0 / refIndex;
			cos = -dot(rIn.direction, rec.normal) / rIn.direction.length();
		}

		if (refract(rIn.direction, oNorm, rri, refracted))
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

	__device__ virtual Vec3 emitted(double u, double v, const Vec3& point) const
	{
		return Vec3(0, 0, 0);
	}
};

class DiffuseLight : public Material
{
public:
	Texture* emitTexture;

	__device__ DiffuseLight(Texture* a) : emitTexture(a) {}

	__device__ virtual bool scatter(const Ray& rIn, const HitRecord& rec, Vec3& attenuation, Ray& scattered, curandState* localRandState) const
	{
		return false;
	}

	__device__ virtual Vec3 emitted(double u, double v, const Vec3& point) const
	{
		return emitTexture->value(u, v, point);
	}
};

class Isotropic : public Material
{
public:
	Texture* albedo;

	__device__ Isotropic(Texture* a) :albedo(a) {}

	__device__ virtual bool scatter(const Ray& rIn, const HitRecord& rec, Vec3& attenuation, Ray& scattered, curandState* localRandState) const
	{
		scattered = Ray(rec.point, randomVecInUnitSphere(localRandState));
		attenuation = albedo->value(rec.u, rec.v, rec.point);
		return true;
	}

	__device__ virtual Vec3 emitted(double u, double v, const Vec3& point) const
	{
		return Vec3(0, 0, 0);
	}
};