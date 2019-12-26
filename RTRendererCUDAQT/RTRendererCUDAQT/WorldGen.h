#pragma once
#include "misc.h"
#include "Hittable.h"
#include "Camera.h"
#include "Material.h"
#include "Sphere.h"
#include "Rectangles.h"
#include "Triangle.h"
#include "Box.h"
#include "Translation.h"
#include "Volumes.h"
#include "HittableList.h"
#include "BVH.h"
#include <fstream>
#include <vector>

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
	float fov = 60.0f;
	*camera = new Camera(MAX_X, MAX_Y, fov, lookfrom, lookat, Vec3(0, 1, 0), aperture, focusDist);
}

#undef RND
#define RND (curand_uniform(randState))
__global__ void createRandScene(Hittable** list, Hittable** world, Camera** camera, unsigned char* texture, int tx, int ty, curandState* randState,
	Vec3 lookat, Vec3 lookfrom, Vec3 vup, float focusDist, float aperture, float fov)
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

	/*
	Vec3 lookfrom(13, 2, 3);
	Vec3 lookat(0, 0, 0);
	Vec3 vup(0, 1, 0);
	float focusDist = (lookfrom - lookat).length() - 4;
	float aperture = 0.1;
	float fov = 30.0;
	*/
	/*
	lookfrom = Vec3(13, 2, 3);
	lookat = Vec3(0, 0, 0);
	vup = Vec3(0, 1, 0);
	*/
	focusDist = (lookfrom - lookat).length() - 4;
	aperture = 0.1;
	fov = 30.0;

	*camera = new Camera(MAX_X, MAX_Y, fov, lookfrom, lookat, vup , aperture, focusDist);
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
//	list[i++] = new Box(Vec3(0, 0, 0), Vec3(165, 165, 165), new Lambertian(new ConstantTexture(0.73, 0.73, 0.73)));
//	list[i++] = new Box(Vec3(0, 0, 0), Vec3(165, 330, 165), new Lambertian(new ConstantTexture(0.73, 0.73, 0.73)));
	list[i++] = new Box(Vec3(0, 0, 0), Vec3(165, 165, 165), new Dielectric(Vec3(1, 1, 1),1.5,0));
	list[i++] = new Box(Vec3(0, 0, 0), Vec3(165, 330, 165), new Dielectric(Vec3(0.9, 0.9, 1), 1.5, 0));

	list[0] = new FlipNorm(list[0]);
	list[3] = new FlipNorm(list[3]);
	list[5] = new FlipNorm(list[5]);

	list[6] = new RotateY(list[6], -18);
	list[6] = new Translate(list[6], Vec3(130, 0, 65));

	list[7] = new RotateY(list[7], 15);
	list[7] = new Translate(list[7], Vec3(265, 0, 295));


	int faces[12] = {
		0,1,2,
		0,1,3,
		0,2,3,
		1,2,3
};
	int nTriangle = 4;
	Vec3 vertexs[4] = {
		Vec3(260,260,0),
		Vec3(300,260,0),
		Vec3(280,294,0),
		Vec3(280,280,60)
	};
	list[i++] = new TriangleMesh(faces, vertexs, 4, 4, new Dielectric(Vec3(0.1, 0.8, 1), 1.5, 0));

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

__global__ void triMeshTest(Hittable** list, Hittable** world, Camera** camera)
{
	printMsg(LogLevel::info, "Using scene: Triangle test.");

	int faces[12] = { 
		0,1,2,
		0,1,3,
		0,2,3,
		1,2,3 };
	int nTriangle = 4;
	Vec3 vertexs[4] = {
		Vec3(0,0,0),
		Vec3(2,0,0),
		Vec3(1,1.7,0),
		Vec3(1,1,1)
	};

	list[0] = new TriangleMesh(faces, vertexs, 4, 4, new Lambertian(new ConstantTexture(0.2, 0.2, 0.8)));

	*world = new HittableList(list, 1);

	Vec3 lookfrom(10, 0, 10);
	Vec3 lookat(0, 0, 0);
	float focusDist = (lookfrom - lookat).length();
	float aperture = 0.05;
	*camera = new Camera(MAX_X, MAX_Y, 60.0f, lookfrom, lookat, Vec3(0, 1, 0), aperture, focusDist);
}

__global__ void meshTest(unsigned char* texture, int tx, int ty, int* faces, Vec3* vertexs, int nface, int nvertex, Hittable** list, Hittable** world)
{
	list[0] = new TriangleMesh(faces, vertexs, nface, nvertex, new Lambertian(new ConstantTexture(0.8, 0.8, 0.8)));
	list[1] = new Sphere(Vec3(1200, 1200, -1200), 300, new DiffuseLight(new ConstantTexture(10, 10, 10)));
	list[2] = new Sphere(Vec3(0, -300, -1), 200, new Lambertian(new ImageTexture(texture, tx, ty)));

	*world = new HittableList(list, 3);
}

__global__ void camInit(Vec3 lookat, Vec3 lookfrom, Vec3 vup, float focusDist, float aperture, float fov, Camera** camera)
{
	*camera = new Camera(MAX_X, MAX_Y, fov, lookfrom, lookat, vup, aperture, focusDist);
}

__host__ void meshTestHost(Hittable** list, Hittable** world, int* allow, std::string fileName)
{
	printMsg(LogLevel::info, "Loading texture...");
	cv::Mat em;
	unsigned char* texture;
	em = cv::imread("earthmap.jpg");
	if (em.rows < 1 || em.cols < 1)
	{
		printMsg(LogLevel::error, "Failed to find Earth texture(earthmap.jpg).");
	}
	else
	{
		printMsg(LogLevel::debug, "Texture loaded.");
	}

	checkCudaErrors(cudaMalloc((void**)&texture, sizeof(unsigned char) * em.rows * em.cols * 3));
	checkCudaErrors(cudaMemcpy(texture, em.data, sizeof(unsigned char) * em.rows * em.cols * 3, cudaMemcpyHostToDevice));

	printMsg(LogLevel::info, "Loading mesh...");
	std::ifstream lowPolyDeer(fileName, std::ifstream::in);
	std::vector<int> f;
	std::vector<Vec3> v;
	while (lowPolyDeer.good() && allow[0] == 1)
	{
		std::string a, b, c, d;
		lowPolyDeer >> a;
		if (a == "v")
		{
			lowPolyDeer >> b >> c >> d;
			v.push_back(Vec3(std::stof(b), std::stof(c), std::stof(d)));
			printMsg(LogLevel::extra, "vertex: %f, %f, %f", std::stof(b), std::stof(c), std::stof(d));
		}
		else if(a == "f")
		{
			lowPolyDeer >> b >> c >> d;
			f.push_back(std::stoi(b));
			f.push_back(std::stoi(c));
			f.push_back(std::stoi(d));
			printMsg(LogLevel::extra, "face: %d, %d, %d", std::stoi(b), std::stoi(c), std::stoi(d));
		}
		else
		{
			lowPolyDeer.ignore(100, '\n');
		}
	}
	printMsg(LogLevel::info, "3D file parsed with %d faces and %d vertexs.", f.size()/3, v.size());
	int* fh = new int[f.size()], *faces;
	Vec3* vh = new Vec3[v.size()], *vertexs;

	checkCudaErrors(cudaMalloc((void**)&faces, sizeof(int) * f.size()));
	checkCudaErrors(cudaMalloc((void**)&vertexs, sizeof(Vec3) * v.size()));

	for (int i = 0; i < f.size(); i++)
	{
		fh[i] = f[i]-1;
	}

	for (int i = 0; i < v.size(); i++)
	{
		vh[i] = v[i];
	}

	checkCudaErrors(cudaMemcpy(faces, fh, sizeof(int) * f.size(), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(vertexs, vh, sizeof(Vec3) * v.size(), cudaMemcpyHostToDevice));

	meshTest <<<1, 1>>> (texture, em.cols, em.rows, faces, vertexs, f.size()/3, v.size(), list, world);
}