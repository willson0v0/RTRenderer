#include <iostream>
#include <opencv/cv.hpp>
#include <float.h>
#include "Vec3.h"
#include "Ray.h"
#include "Sphere.h"
#include "Hittable.h"
#include "Material.h"
#include "HittableList.h"
#include "Camera.h"
#include "misc.h"

constexpr auto AA = 128;

constexpr auto x = 1024;
constexpr auto y = 768;

Hittable* randomScene()
{
	int n = 501;
	Hittable** list = new Hittable * [n];
	list[0] = new Sphere(0, -10000, 0, 10000, new Metal(0.2, 0.8, 0.5, 0.02));
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
							randD() * randD(),
							randD() * randD(),
							randD() * randD()
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

	list[i++] = new Sphere(4, 1, 0, 1, new Lambertian(0.3, 0.3, 0.8));
	list[i++] = new Sphere(0, 1, 0, 1, new Metal(0.8, 0.2, 0.8, 0));
	list[i++] = new Sphere(-4, 1, 0, 1, new dielectric(1, 1, 1, 1.5, 0));
	list[i++] = new Sphere(-4, 1, 0, -0.75, new dielectric(1, 1, 1, 1.5, 0));

	return new HittableList(list, i);
}


Vec3 getColor(const Ray& r, Hittable* world, int depth)
{
	hitRecord rec;
	if (world->hit(r, 0.0001, 1.79769e308, rec))
	{
		Ray scattered;
		Vec3 attenuation;
		if (depth < 50 && rec.matPtr->scatter(r, rec, attenuation, scattered))
		{
			return attenuation * getColor(scattered, world, depth + 1);
		}
		else
		{
			return Vec3(0, 0, 0);
		}
	}

	Vec3 unitDir = unitVector(r.direction());
	double p = 0.5 * (unitDir.y() + 1);
	return (1.0 - p) * Vec3(1, 1, 1) + p * Vec3(0.5, 0.7, 1);
}

int main()
{
	cv::Mat M(y, x, CV_8UC3, cv::Scalar(0, 0, 0));

	Hittable* world = randomScene();

	Vec3 lookFrom(-13, 2, 3);
	Vec3 lookAt(0, 0, 0);
	Vec3 vup(0, 1, 0);

	Camera cam(x, y, 20, lookFrom, lookAt, vup, 0.1, (lookFrom - lookAt).length());

	cv::namedWindow("wow", cv::WINDOW_AUTOSIZE);

	for (int iter = 0; iter < AA; iter++)
	{
#pragma omp parallel for
		for (int i = 0; i < y; i++)
		{
#pragma omp parallel for
			for (int j = 0; j < x; j++)
			{
				cv::Vec3b prev = M.at<cv::Vec3b>(y - i - 1, j);
				Vec3 pix(double(prev[2]) / 255.9, double(prev[1]) / 255.9, double(prev[0]) / 255.9);
				pix *= pix; //gamma correction
				pix *= (iter*4);
				for (int k = 0; k < 4; k++)
				{
					double u = (double(j) + randD()) / double(x);
					double v = (double(i) + randD()) / double(y);
					Ray r = cam.getRay(u, v);
					pix += getColor(r, world, 0);
				}
				M.at<cv::Vec3b>(y - i - 1, j) = (pix / (iter * 4 + 4)).toCVPix();
			}
		}
		cv::imshow("wow", M);
		cv::waitKey(1);
	}

	cv::waitKey();
	return 0;
}