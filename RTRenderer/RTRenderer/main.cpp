#include <iostream>
#include <opencv/cv.hpp>
#include <float.h>
#include "Vec3.h"
#include "Ray.h"
#include "Sphere.h"
#include "Hittable.h"
#include "HittableList.h"

Sphere a(0, 0, -5, 3);

Vec3 getColor(const Ray& r, Hittable* world)
{
	hitRecord rec;
	if (world->hit(r, 0.0, 1.79769e308, rec))
	{
		return unitVector(rec.norm + Vec3(1, 1, 1));
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

	Vec3 llc(-double(x) / double(y) * 5.0, -5, -5);
	Vec3 horizonal(double(x)/double(y)*10.0, 0, 0);
	Vec3 vertical(0, 10, 0);
	Vec3 origin(0, 0, 0);

	Hittable* list[2];
	list[0] = new Sphere(0, 0, -5, 3);
	list[1] = new Sphere(0, -504, -5, 500);
	Hittable* world = new HittableList(list, 2);

	for (int i = 0; i < y; i++)
	{
		for (int j = 0; j < x; j++)
		{
			double u = double(j) / double(x);
			double v = double(i) / double(y);
			Ray r(origin, llc + u * horizonal + v * vertical);
			Vec3 pix = getColor(r, world);

			M.at<cv::Vec3b>(y - i - 1, j) = pix.toCVPix();
		}
	}

	cv::namedWindow("wow", cv::WINDOW_AUTOSIZE);
	cv::imshow("wow", M);

	cv::waitKey();

	return 0;
}