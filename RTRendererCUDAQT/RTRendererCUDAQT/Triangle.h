#pragma once
#include "consts.h"
#include "Hittable.h"
#include "BVH.h"

class TriangleMesh : public Hittable
{
public:
	class Triangle :public Hittable
	{
	public:
		TriangleMesh* mesh;
		int* vertexIndex;
		int index;

		__device__ Triangle(int* vi, int indexInMesh, TriangleMesh* m):mesh(m)
		{
			vertexIndex = new int[3];
			vertexIndex[0] = vi[0];
			vertexIndex[1] = vi[1];
			vertexIndex[2] = vi[2];
			index = indexInMesh;
		}

		__device__ virtual bool hit(const Ray& r, float tMin, float tMax, HitRecord& rec, curandState* localRandState) const
		{
			Vec3 e1 = mesh->vertex[vertexIndex[1]] - mesh->vertex[vertexIndex[0]];
			Vec3 e2 = mesh->vertex[vertexIndex[2]] - mesh->vertex[vertexIndex[0]];
			Vec3 p = cross(r.direction, e2);
			float det = dot(e1, p), u, v, t;
			if (det > -0.0001 && det < 0.0001) return false;
			Vec3 T;
			if (det > 0)
			{
				T = r.origin - mesh->vertex[vertexIndex[0]];
			}
			else
			{
				T = mesh->vertex[vertexIndex[0]] - r.origin;
				det = -det;
			}
			u = dot(T, p);
			if (u < 0 || u > det) return false;
			Vec3 q = cross(T, e1);

			v = dot(r.direction, q);
			if (v<0 || u + v > det) return false;

			t = dot(e2, q)/det;

			if (t > tMin && t < tMax) {
				rec.t = t;
				rec.point = r.pointAtParam(t);
				rec.normal = cross(e1, e2);
				if (dot(rec.normal, r.direction) > 0)
				{
					rec.normal = Vec3(0, 0, 0) - rec.normal;
				}
				rec.matPtr = mesh->matPtr;
				return true;
			}
			return false;
		}

		__device__ virtual bool boundingBox(AABB& box) const
		{
			Vec3 s(FLT_MAX, FLT_MAX, FLT_MAX);
			Vec3 b(-FLT_MAX, -FLT_MAX, -FLT_MAX);

			for (int i = 0; i < 3; i++)
			{
				for (int j = 0; j < 3; j++)
				{
					if (mesh->vertex[vertexIndex[i]].e[j] < s.e[j]) s.e[j] = mesh->vertex[vertexIndex[i]].e[j];
					if (mesh->vertex[vertexIndex[i]].e[j] > b.e[j]) b.e[j] = mesh->vertex[vertexIndex[i]].e[j];
				}
			}
			box = AABB(s, b);
			return true;
		}
	};

	Vec3* vertex;
	Material* matPtr;
	Hittable** triangles;

#ifdef USE_BVH
	BVH* bvh;
#endif

	int nTriangles;
	int nVertex;

	double* u;
	double* v;

	__device__ TriangleMesh(int* triVerList, Vec3* vr, int nt, int nv, Material* m, curandState* localRandState)
		:nTriangles(nt), nVertex(nv), matPtr(m)
	{
		triangles = new Hittable * [nTriangles];
		for (int i = 0; i < nTriangles; i++)
		{
			triangles[i] = new Triangle(&triVerList[i * 3], i, this);
		}

		vertex = new Vec3[nVertex];
		for (int i = 0; i < nVertex; i++)
		{
			vertex[i] = vr[i];
		}
#ifdef USE_BVH
		bvh = new BVH(triangles, nTriangles, localRandState);
#endif
	}

	__device__ virtual bool hit(const Ray& r, float tMin, float tMax, HitRecord& rec, curandState* localRandState) const
	{
#ifdef USE_BVH
		return bvh->hit(r, tMin, tMax, rec, localRandState);
#else
		HitRecord tRec;
		bool hitAny = false;
		float closest = tMax;
		for (int i = 0; i < nTriangles; i++)
		{
			if (triangles[i]->hit(r, tMin, closest, tRec, localRandState))
			{
				hitAny = true;
				closest = tRec.t;
				rec = tRec;
			}
		}
		return hitAny;
#endif
	}

	__device__ virtual bool boundingBox(AABB& box) const
	{
#ifdef USE_BVH
		return bvh->boundingBox(box);
#else
		if (nTriangles < 1) return false;
		AABB temp;
		if (!triangles[0]->boundingBox(temp))
		{
			return false;
		}
		else
		{
			box = temp;
		}
		for (int i = 1; i < nTriangles; i++)
		{
			if (triangles[i]->boundingBox(temp))
			{
				box = surroundingBox(temp, box);
			}
			else return false;
		}
		return true;
#endif
	}
};