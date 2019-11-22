#pragma once
#include "misc.h"
#include "Hittable.h"
#include "Camera.h"
#include "Material.h"
#include "Sphere.h"
#include "Rectangles.h"
#include "HittableList.h"
#include "BVH.h"

__global__ void generateCamera(Camera** cameraPtr, Vec3 lookFrom, Vec3 lookAt, Vec3 vup, double vfov,double aperture, double focusDist = -1)
{
	if (focusDist == -1) focusDist = (lookAt - lookFrom).length();
	*cameraPtr = new Camera(MAX_X, MAX_Y, vfov, lookFrom, lookAt, vup, aperture, focusDist);
}

__global__ void createWorld1(Hittable** list, Hittable** world, Camera** camera)
{
	list[0] = new Sphere(Vec3(-4, 1, 0), 1.0, new Lambertian(new ConstantTexture(Vec3(0.4, 0.2, 0.1))));
	list[1] = new Sphere(Vec3(0, 1, 0), 1.0, new Metal(Vec3(0.4, 0.5, 0.9), 0));
	list[2] = new Sphere(Vec3(4, 1, 0), 1.0, new Dielectric(Vec3(1, 1, 1), 1.5, 0));
	list[3] = new Sphere(Vec3(4, 1, 0), -0.75, new Dielectric(Vec3(1, 1, 1), 1.5, 0));
	list[4] = new Sphere(Vec3(0, -10000, -1), 10000, new Metal(Vec3(0.8, 0.7, 0.6), 0));

	*world = new HittableList(list, 5);

	Vec3 lookfrom(13, 2, 3);
	Vec3 lookat(0, 0, 0);
	double focusDist = (lookfrom - lookat).length() - 4;
	double aperture = 0.1;
	*camera = new Camera(MAX_X, MAX_Y, 30.0f, lookfrom, lookat, Vec3(0, 1, 0), aperture, focusDist);
}

#define RND (curand_uniform(randState))
__global__ void createRandScene(Hittable** list, Hittable** world, Camera** camera, unsigned char* texture, int tx, int ty, curandState* randState)
{
	curand_init(clock(), 0, 0, randState);
	Perlin::initPerlin(randState);

	int i = 0;
	list[i++] = new Sphere(Vec3(0, -10000.0, -1), 10000, new Lambertian(
		new CheckerTexture(
			new ConstantTexture(Vec3(0.5, 0.5, 0.5)), 
			new ConstantTexture(Vec3(0.9, 0.9, 0.9))
		)
	));
	list[i++] = new Sphere(Vec3(4, 1, 0), 1.0, new Dielectric(Vec3(1, 1, 1), 1.5, 0));
	list[i++] = new Sphere(Vec3(4, 1, 0), -0.75, new Dielectric(Vec3(1, 1, 1), 1.5, 0));
	list[i++] = new Sphere(Vec3(4, 1, 0), 0.5, new Lambertian(new ImageTexture(texture, tx, ty)));
	list[i++] = new Sphere(Vec3(0, 1, 0), 1.0, new Lambertian(new NoiseTexture(4)));
	list[i++] = new Sphere(Vec3(-4, 1, 0), 1.0, new Metal(Vec3(0.7, 0.6, 0.5), 0.0));

	list[i++] = new Sphere(Vec3(-5, 3, 2), 1, new DiffuseLight(new ConstantTexture(Vec3(0.8, 0.2, 0.8))));
	list[i++] = new FlipNorm(new RectXY(3, 5, 1, 3, -3, new DiffuseLight(new ConstantTexture(Vec3(0.2, 0.8, 0.8)))));
	
	for (int a = -11; a < 11; a++) {
		for (int b = -11; b < 11; b++) {
			double choose_mat = RND;
			double radius = 0.1 * RND + 0.1;
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

	curand_init(2019, 0, 0, randState);

	*world = new HittableList(list, 22*22+8);
	//*world = new BVH(list, 22*22+8);

	Vec3 lookfrom(13, 2, 3);
	Vec3 lookat(0, 0, 0);
	double focusDist = (lookfrom - lookat).length() - 4;
	double aperture = 0.1;
	*camera = new Camera(MAX_X, MAX_Y, 30, lookfrom, lookat, Vec3(0, 1, 0), aperture, focusDist);
}