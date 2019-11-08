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

Vec3 getColor(const Ray& r, Hittable* world, int depth)
{
	hitRecord rec;
	if (world->hit(r, 0.001, 1.79769e308, rec))
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
	int x = 1920;
	int y = 1080;
	cv::Mat M(y, x, CV_8UC3, cv::Scalar(0, 0, 255));

	Hittable* list[4];
	list[0] = new Sphere(-6, 0, -15, 3, new Lambertian(0.5, 0.2, 0.7));
	list[1] = new Sphere(0, -503, -15, 500, new Metal(0.2, 0.8, 0.5));
	list[2] = new Sphere(0, 0, -15, 3, new Metal(0.8, 0.8, 0.2));
	list[3] = new Sphere(6, 0, -15, 3, new Lambertian(0.3, 0.3, 0.8));
	Hittable* world = new HittableList(list, 4);
	Camera cam(x, y);

	double bias[2][4] = { {0.25, 0.75, 0.25, 0.75}, {0.25, 0.25, 0.75, 0.75} };

	for (int i = 0; i < y; i++)
	{
		for (int j = 0; j < x; j++)
		{
			Vec3 pix(0,0,0);
			for (int k = 0; k < 4; k++)
			{
				double u = (double(j) + bias[0][k]) / double(x);
				double v = (double(i) + bias[1][k]) / double(y);
				Ray r = cam.getRay(u, v);
				pix += getColor(r, world, 0);
			}
			pix /= 4;
			M.at<cv::Vec3b>(y - i - 1, j) = pix.toCVPix();
		}
	}

	cv::namedWindow("wow", cv::WINDOW_AUTOSIZE);
	cv::imshow("wow", M);

	cv::waitKey();

	return 0;
}