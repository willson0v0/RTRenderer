#pragma once
#include "misc.h"
#include "Hittable.h"
#include "Camera.h"
#include "Material.h"
#include "Sphere.h"
#include "Rectangles.h"
#include "Box.h"
#include "Translation.h"
#include "Volumes.h"
#include "HittableList.h"
#include "BVH.h"

__global__ void generateCamera(Camera** cameraPtr, Vec3 lookFrom, Vec3 lookAt, Vec3 vup, float vfov,float aperture, float focusDist = -1)
{
	if (focusDist == -1) focusDist = (lookAt - lookFrom).length();
	*cameraPtr = new Camera(MAX_X, MAX_Y, vfov, lookFrom, lookAt, vup, aperture, focusDist);
}

__global__ void createWorld1(Hittable** list, Hittable** world, Camera** camera, curandState* randState)
{
	list[0] = new Sphere(Vec3(-4, 1, 0), 1.0, new Lambertian(new ConstantTexture(Vec3(0.4, 0.2, 0.1))));
	list[1] = new Sphere(Vec3(0, 1, 0), 1.0, new Metal(Vec3(0.4, 0.5, 0.9), 0));
	list[2] = new Sphere(Vec3(4, 1, 0), 1.0, new Dielectric(Vec3(1, 1, 1), 1.5, 0));
	list[3] = new Sphere(Vec3(4, 1, 0), -0.75, new Dielectric(Vec3(1, 1, 1), 1.5, 0));
	list[4] = new Sphere(Vec3(0, -10000, -1), 10000, new Metal(Vec3(0.8, 0.7, 0.6), 0));

#ifdef USE_BVH
	* world = new BVH(list, 5, randState);
#else
	* world = new HittableList(list, 5);
#endif

	Vec3 lookfrom(13, 2, 3);
	Vec3 lookat(0, 0, 0);
	float focusDist = (lookfrom - lookat).length() - 4;
	float aperture = 0.1;
	*camera = new Camera(MAX_X, MAX_Y, 30.0f, lookfrom, lookat, Vec3(0, 1, 0), aperture, focusDist);
}

__global__ void createCheckerTest(Hittable** list, Hittable** world, Camera** camera, curandState* randState)
{
	list[0] = new Sphere(Vec3(0, -50.0, -1), 50, new Lambertian(
		new CheckerTexture(
			new ConstantTexture(Vec3(0.3, 0.5, 0.9)),
			new ConstantTexture(Vec3(0.9, 0.9, 0.9))
		)
	));
	list[1] = new Sphere(Vec3(0, 50.0, -1), 50, new Lambertian(
		new CheckerTexture(
			new ConstantTexture(Vec3(0.9, 0.9, 0.9)),
			new ConstantTexture(Vec3(0.5, 0.3, 0.9))
		)
	));

#ifdef USE_BVH
	* world = new BVH(list, 2, randState);
#else
	* world = new HittableList(list, 2);
#endif

	Vec3 lookfrom(10, 0, 10);
	Vec3 lookat(0, 0, 0);
	float focusDist = (lookfrom - lookat).length();
	float aperture = 0.05;
	*camera = new Camera(MAX_X, MAX_Y, 60.0f, lookfrom, lookat, Vec3(0, 1, 0), aperture, focusDist);
}

#undef RND
#define RND (curand_uniform(randState))
__global__ void createRandScene(Hittable** list, Hittable** world, Camera** camera, unsigned char* texture, int tx, int ty, curandState* randState)
{
	printMsg(LogLevel::info, "Using scene: Random balls.");

	curand_init(clock(), 0, 0, randState);
	printMsg(LogLevel::debug, "curandInit complete.");

	initPerlin(randState);
	printMsg(LogLevel::debug, "Noise Generation complete.");

	int i = 0;
	list[i++] = new Sphere(Vec3(0, -10000.0, -1), 10000, new Lambertian(
			new CheckerTexture(
			new ConstantTexture(Vec3(0.5, 0.5, 0.5)), 
			new ConstantTexture(Vec3(0.9, 0.9, 0.9))
		)
	));
	list[i++] = new Sphere(Vec3(4, 1, 0), 1.0, new Dielectric(Vec3(1, 1, 1), 1.5, 0));
	list[i++] = new FlipNorm(new Sphere(Vec3(4, 1, 0), 0.75, new Dielectric(Vec3(1, 1, 1), 1.5, 0)));
	list[i++] = new Sphere(Vec3(4, 1, 0), 0.5, new Lambertian(new ImageTexture(texture, tx, ty)));
	list[i++] = new Sphere(Vec3(0, 1, 0), 1.0, new Lambertian(new NoiseTexture(4)));
	list[i++] = new Sphere(Vec3(-4, 1, 0), 1.0, new Metal(Vec3(0.7, 0.6, 0.5), 0.0));

	list[i++] = new Sphere(Vec3(-5, 3, 2), 1, new DiffuseLight(new ConstantTexture(Vec3(10, 5, 10))));
	list[i++] = new RectXY(3, 5, 1, 3, -3, new DiffuseLight(new ConstantTexture(Vec3(5, 10, 5))));
	
	for (int a = -11; a < 11; a++) {
		for (int b = -11; b < 11; b++) {
			float choose_mat = RND;
			float radius = 0.1 * RND + 0.1;
			Vec3 center(RND*0.9 + a, radius, RND*0.9 + b);
			if (choose_mat < 0.4f) {
				list[i++] = new Sphere(center, radius,
					new Lambertian(new ConstantTexture(Vec3(RND * RND, RND * RND, RND * RND))));
			}
			else if (choose_mat < 0.7f) {
				list[i++] = new Sphere(center, radius,
					new Metal(Vec3(0.5 * (1.0 + RND), 0.5 * (1.0 + RND), 0.5 * (1.0 + RND)), 0.4 * RND));
			}
			else {
				list[i++] = new Sphere(center, radius, new Dielectric(Vec3(0.8+0.2*RND, 0.8 + 0.2 * RND, 0.8 + 0.2 * RND), 1.5, 0.1*RND));
			}
		}
	}

	printMsg(LogLevel::debug, "Scene generation complete.");

	curand_init(2019, 0, 0, randState);

#ifdef USE_BVH
	* world = new BVH(list, i, randState);
#else
	*world = new HittableList(list, 22*22+8);
#endif
	printMsg(LogLevel::debug, "Scene Loaded.");

	Vec3 lookfrom(13, 2, 3);
	Vec3 lookat(0, 0, 0);
	float focusDist = (lookfrom - lookat).length() - 4;
	float aperture = 0.1;
	float fov = 30.0;
	*camera = new Camera(MAX_X, MAX_Y, fov, lookfrom, lookat, Vec3(0, 1, 0), aperture, focusDist);
}

__global__ void createCornellBox(Hittable** list, Hittable** world, Camera** camera, curandState* randState)
{
	curand_init(clock(), 0, 0, randState);
	int i = 0;

	list[i++] = new RectYZ(000, 555, 000, 555, 555, new Lambertian(new ConstantTexture(0.12, 0.45, 0.15)));
	list[i++] = new RectYZ(000, 555, 000, 555, 000,	new Lambertian(new ConstantTexture(0.65, 0.05, 0.05)));
	list[i++] = new RectXZ(213, 343, 227, 332, 554, new DiffuseLight(new ConstantTexture(20, 20, 20)));
	list[i++] = new RectXZ(000, 555, 000, 555, 555, new Lambertian(new ConstantTexture(0.73, 0.73, 0.73)));
	list[i++] = new RectXZ(000, 555, 000, 555, 000, new Lambertian(new ConstantTexture(0.73, 0.73, 0.73)));
	list[i++] = new RectXY(000, 555, 000, 555, 555, new Lambertian(new ConstantTexture(0.73, 0.73, 0.73)));

	list[i++] = new Box(Vec3(0, 0, 0), Vec3(165, 165, 165), new Lambertian(new ConstantTexture(0.73, 0.73, 0.73)));
	list[i++] = new Box(Vec3(0, 0, 0), Vec3(165, 330, 165), new Lambertian(new ConstantTexture(0.73, 0.73, 0.73)));

	list[0] = new FlipNorm(list[0]);
	list[3] = new FlipNorm(list[3]);
	list[5] = new FlipNorm(list[5]);

	list[6] = new RotateY(list[6], -18);
	list[6] = new Translate(list[6], Vec3(130, 0, 65));

	list[7] = new RotateY(list[7], 15);
	list[7] = new Translate(list[7], Vec3(265, 0, 295));


#ifdef USE_BVH
	* world = new BVH(list, i, randState);
#else
	* world = new HittableList(list, i);
#endif

	Vec3 lookfrom(278, 278, -800);
	Vec3 lookat(278, 278, 0);
	float focusDist = (lookfrom - lookat).length();
	float aperture = 0;
	*camera = new Camera(MAX_X, MAX_Y, 40, lookfrom, lookat, Vec3(0, 1, 0), aperture, focusDist);
}


__global__ void createCornellSmoke(Hittable** list, Hittable** world, Camera** camera, curandState* randState)
{
	printMsg(LogLevel::info, "Using scene: Cornell Smoke.");

	curand_init(clock(), 0, 0, randState);
	int i = 0;

	list[i++] = new RectYZ(000, 555, 000, 555, 555, new Lambertian(new ConstantTexture(0.12, 0.45, 0.15)));
	list[i++] = new RectYZ(000, 555, 000, 555, 000, new Lambertian(new ConstantTexture(0.65, 0.05, 0.05)));
	list[i++] = new RectXZ(113, 443, 127, 432, 554, new DiffuseLight(new ConstantTexture(7, 7, 7)));
	list[i++] = new RectXZ(000, 555, 000, 555, 555, new Lambertian(new ConstantTexture(0.73, 0.73, 0.73)));
	list[i++] = new RectXZ(000, 555, 000, 555, 000, new Lambertian(new ConstantTexture(0.73, 0.73, 0.73)));
	list[i++] = new RectXY(000, 555, 000, 555, 555, new Lambertian(new ConstantTexture(0.73, 0.73, 0.73)));

	list[i++] = new Box(Vec3(0, 0, 0), Vec3(165, 165, 165), new Lambertian(new ConstantTexture(0.73, 0.73, 0.73)));
	list[i++] = new Box(Vec3(0, 0, 0), Vec3(165, 330, 165), new Lambertian(new ConstantTexture(0.73, 0.73, 0.73)));

	list[i++] = new Sphere(Vec3(450, 70, 50), 70, new Dielectric(Vec3(1, 1, 1), 1.5, 0));

	list[i++] = new ConstantMedium(list[8], 0.05, new ConstantTexture(Vec3(0.2, 0.4, 0.9)));

	list[0] = new FlipNorm(list[0]);
	list[3] = new FlipNorm(list[3]);
	list[5] = new FlipNorm(list[5]);

	list[6] = new RotateY(list[6], -18);
	list[6] = new Translate(list[6], Vec3(130, 0, 65));
	list[6] = new ConstantMedium(list[6], 0.01, new ConstantTexture(Vec3(1, 1, 1)));

	list[7] = new RotateY(list[7], 15);
	list[7] = new Translate(list[7], Vec3(265, 0, 295));
	list[7] = new ConstantMedium(list[7], 0.01, new ConstantTexture(Vec3(0, 0, 0)));


#ifdef USE_BVH
	* world = new BVH(list, i, randState);
#else
	* world = new HittableList(list, i);
#endif

	Vec3 lookfrom(278, 278, -800);
	Vec3 lookat(278, 278, 0);
	float focusDist = (lookfrom - lookat).length();
	float aperture = 0;
	*camera = new Camera(MAX_X, MAX_Y, 40, lookfrom, lookat, Vec3(0, 1, 0), aperture, focusDist);
}