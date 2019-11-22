#pragma once

#include "consts.h"

class Vec3
{
public:
	double e[3];

	__host__ __device__ Vec3() 
	{
		e[0] = 0;
		e[1] = 0;
		e[2] = 0;
	}

	__host__ __device__ Vec3(double e0, double e1, double e2) 
	{
		e[0] = e0;
		e[1] = e1;
		e[2] = e2;
	}

	__host__ __device__ inline const Vec3& operator+() const { return *this; }
	__host__ __device__ inline Vec3 operator-() const { return Vec3(-e[0], -e[1], -e[2]); }
	__host__ __device__ inline Vec3& operator+=(const Vec3& v2);
	__host__ __device__ inline Vec3& operator-=(const Vec3& v2);
	__host__ __device__ inline Vec3& operator*=(const Vec3& v2);
	__host__ __device__ inline Vec3& operator/=(const Vec3& v2);
	__host__ __device__ inline Vec3& operator*=(const double v2);
	__host__ __device__ inline Vec3& operator/=(const double v2);


	__host__ __device__ inline double squaredLength() const { return e[0] * e[0] + e[1] * e[1] + e[2] * e[2]; }
	__host__ __device__ inline double length() const { return sqrt(squaredLength()); }
	__host__ __device__ inline void makeUnitVector();

	__host__ __device__ inline void Vec3::writeFrameBuffer(int i, int j, double* fBuffer);
	__host__ __device__ inline void Vec3::readFrameBuffer(int i, int j, double* fBuffer);
};

inline std::istream& operator>>(std::istream& is, Vec3& t)
{
	is >> t.e[0] >> t.e[1] >> t.e[2];
	return is;
}

inline std::ostream& operator<<(std::ostream& os, Vec3& t)
{
	os << t.e[0] << " " << t.e[1] << " " << t.e[2];
	return os;
}

__host__ __device__ inline void Vec3::makeUnitVector()
{
	double k = 1.0 / squaredLength();
	e[0] *= k;
	e[1] *= k;
	e[2] *= k;
}

__host__ __device__ inline Vec3 operator+(const Vec3& v1, const Vec3& v2)
{
	return Vec3(
		v1.e[0] + v2.e[0],
		v1.e[1] + v2.e[1],
		v1.e[2] + v2.e[2]
	);
}

__host__ __device__ inline Vec3 operator-(const Vec3& v1, const Vec3& v2)
{
	return Vec3(
		v1.e[0] - v2.e[0],
		v1.e[1] - v2.e[1],
		v1.e[2] - v2.e[2]
	);
}

__host__ __device__ inline Vec3 operator*(const Vec3& v1, const Vec3& v2)
{
	return Vec3(
		v1.e[0] * v2.e[0],
		v1.e[1] * v2.e[1],
		v1.e[2] * v2.e[2]
	);
}

__host__ __device__ inline Vec3 operator/(const Vec3& v1, const Vec3& v2)
{
	return Vec3(
		v1.e[0] / v2.e[0],
		v1.e[1] / v2.e[1],
		v1.e[2] / v2.e[2]
	);
}

__host__ __device__ inline Vec3 operator*(const Vec3& v1, double t)
{
	return Vec3(
		v1.e[0] * t,
		v1.e[1] * t,
		v1.e[2] * t
	);
}

__host__ __device__ inline Vec3 operator*(double t, const Vec3& v1)
{
	return v1*t;
}

__host__ __device__ inline Vec3 operator/(const Vec3& v1, double t)
{
	return Vec3(
		v1.e[0] / t,
		v1.e[1] / t,
		v1.e[2] / t
	);
}


__host__ __device__ inline double dot(const Vec3& v1, const Vec3& v2)
{
	return v1.e[0] * v2.e[0] +
		   v1.e[1] * v2.e[1] +
		   v1.e[2] * v2.e[2];
}

__host__ __device__ inline Vec3 cross(const Vec3& v1, const Vec3& v2)
{
	return Vec3(
		(v1.e[1] * v2.e[2] - v1.e[2] * v2.e[1]),
		(v1.e[2] * v2.e[0] - v1.e[0] * v2.e[2]),
		(v1.e[0] * v2.e[1] - v1.e[1] * v2.e[0])
	);
}


__host__ __device__ inline Vec3& Vec3::operator+=(const Vec3& v2)
{
	e[0] += v2.e[0];
	e[1] += v2.e[1];
	e[2] += v2.e[2];
	return *this;
}

__host__ __device__ inline Vec3& Vec3::operator-=(const Vec3& v2)
{
	e[0] -= v2.e[0];
	e[1] -= v2.e[1];
	e[2] -= v2.e[2];
	return *this;
}

__host__ __device__ inline Vec3& Vec3::operator*=(const Vec3& v2)
{
	e[0] *= v2.e[0];
	e[1] *= v2.e[1];
	e[2] *= v2.e[2];
	return *this;
}

__host__ __device__ inline Vec3& Vec3::operator/=(const Vec3& v2)
{
	e[0] /= v2.e[0];
	e[1] /= v2.e[1];
	e[2] /= v2.e[2];
	return *this;
}

__host__ __device__ inline Vec3& Vec3::operator*=(const double t)
{
	e[0] *= t;
	e[1] *= t;
	e[2] *= t;
	return *this;
}

__host__ __device__ inline Vec3& Vec3::operator/=(const double t)
{
	e[0] /= t;
	e[1] /= t;
	e[2] /= t;
	return *this;
}

__host__ __device__ inline Vec3 unitVector(Vec3 v)
{
	return v / v.length();
}

__host__ __device__ inline void Vec3::writeFrameBuffer(int i, int j, double* fBuffer)
{
	int index = (MAX_Y - j - 1) * MAX_X * 3 + i * 3; // Find location in frame buffer.

	fBuffer[index] = e[2];
	fBuffer[index + 1] = e[1];
	fBuffer[index + 2] = e[0];
}

__host__ __device__ inline void Vec3::readFrameBuffer(int i, int j, double* fBuffer)
{
	int index = (MAX_Y - j - 1) * MAX_X * 3 + i * 3; // Find location in frame buffer.

	e[2] = fBuffer[index];
	e[1] = fBuffer[index+1];
	e[0] = fBuffer[index+2];
}