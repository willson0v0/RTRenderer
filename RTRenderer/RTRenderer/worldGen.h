#pragma once

#include "Material.h"
#include "Hittable.h"
#include "HittableList.h"
#include "Sphere.h"
#include "Texture.h"

Hittable* randomScene()
{
	cv::Mat map = cv::imread("earthmap.jpg");

	int n = 501;
	Hittable** list = new Hittable * [n];
	Texture* checker = new CheckerTexture(new constantTexture(0.1, 0.3, 0.2), new constantTexture(0.9, 0.9, 0.9));
	list[0] = new Sphere(0, -1000, 0, 1000, new Lambertian(checker));
	int i = 1;
	for (int a = -11; a < 11; a++) {
		for (int b = -11; b < 11; b++) {
			double randMaterial = randD();
			double r = 0.1 + 0.1 * randD();
			Vec3 center(a + 0.9 * randD(), r, b + 0.9 * randD());
			if ((center - Vec3(4, 0.2, 0)).length() > 0.9) {
				if (randMaterial < 0.4) {  // diffuse
					list[i++] = new Sphere(center, r,
						new Lambertian(
							new constantTexture(
								randD() * randD(),
								randD() * randD(),
								randD() * randD()
							)
						)
					);
				}
				else if (randMaterial < 0.7) { // metal
					list[i++] = new Sphere(center, r,
						new Metal(
							0.5 * (1 + randD()),
							0.5 * (1 + randD()),
							0.5 * (1 + randD()),
							0.1 * randD()
						)
					);
				}
				else {  // glass
					list[i++] = new Sphere(center, r,
						new dielectric(
							0.2 * (4 + randD()),
							0.2 * (4 + randD()),
							0.2 * (4 + randD()),
							1.5,
							0.2 * randD()
						)
					);
				}
			}
		}
	}
	list[i++] = new Sphere(-4, 1, 0, 1, new Lambertian(new NoiseTexture(4)));
	list[i++] = new Sphere(0, 1, 0, 1, new Metal(0.8, 0.2, 0.8, 0));
	list[i++] = new Sphere(4, 1, 0, 1, new dielectric(1, 1, 1, 1.5, 0));
	list[i++] = new Sphere(4, 1, 0, -0.75, new dielectric(1, 1, 1, 1.5, 0));
	list[i++] = new Sphere(4, 1, 0, 0.5, new Lambertian(new ImageTexture(map, map.cols, map.rows)));

	return new HittableList(list, i);
}

Hittable* twoPerlinSpheres()
{
	Texture* perlinTexture = new NoiseTexture(4);
	Hittable** list = new Hittable * [2];
	list[0] = new Sphere(0, -1000, 0, 1000, new Lambertian(perlinTexture));
	list[1] = new Sphere(0, 2, 0, 2, new Lambertian(perlinTexture));
	return new HittableList(list, 2);
}

Hittable* worldMap()
{
	cv::Mat map = cv::imread("earthmap.jpg");

	Hittable** list = new Hittable * [2];
	list[0] = new Sphere(0, -1000, 0, 1000, new Metal(0.8, 0.2, 0.8, 0.1));
	list[1] = new Sphere(0, 2, 0, 2, new Lambertian(new ImageTexture(map, map.cols, map.rows)));
	return new HittableList(list, 2);
}