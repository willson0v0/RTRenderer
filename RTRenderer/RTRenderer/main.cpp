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
#include "worldGen.h"

constexpr auto AA = 65535;

constexpr auto x = 1024;
constexpr auto y = 768;


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
			//return attenuation;
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

//	Hittable* world = twoPerlinSpheres();
	Hittable* world = randomScene();
//	Hittable* world = worldMap();

	Vec3 lookFrom(13, 2, 3);
	Vec3 lookAt(0, 0, 0);
	Vec3 vup(0, 1, 0);

	Camera cam(x, y, 20, lookFrom, lookAt, vup, 0.1, (lookFrom - lookAt).length() - 4);

	cv::namedWindow("wow", cv::WINDOW_AUTOSIZE);

	for (int iter = 0; iter < AA; iter++)
	{
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
			int progress = 100.0*(double(i+1) / double(y+1));
			if (!(i % 10))
			{
				system("cls");
				std::cout << "Iter No." << iter << std::endl;
				std::cout << "Progress: |" << std::string(progress, '=') << ">" << std::string(100 - progress + 1, ' ') << "| " << progress << "%";
				cv::imshow("wow", M);
				cv::waitKey(1);
			}
		}
	}

	cv::waitKey();
	return 0;
}