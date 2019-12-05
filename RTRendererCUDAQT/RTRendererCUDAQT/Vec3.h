#pragma once

#include "consts.h"

class Vec3
{
public:
	float e[3];

	__host__ __device__ Vec3() 
	{
		e[0] = 0;
		e[1] = 0;
		e[2] = 0;
	}

	__host__ __device__ Vec3(float e0, float e1, float e2) 
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
	__host__ __device__ inline Vec3& operator*=(const float v2);
	__host__ __device__ inline Vec3& operator/=(const float v2);
	__host__ __device__ inline bool operator==(const Vec3& v2);


	__host__ __device__ inline float squaredLength() const { return e[0] * e[0] + e[1] * e[1] + e[2] * e[2]; }
	__host__ __device__ inline float length() const { return sqrt(squaredLength()); }
	__host__ __device__ inline void makeUnitVector();

	__host__ __device__ inline void Vec3::writeFrameBuffer(int i, int j, float* fBuffer);
	__host__ __device__ inline void Vec3::readFrameBuffer(int i, int j, float* fBuffer);
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
	float k = 1.0 / squaredLength();
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

__host__ __device__ inline Vec3 operator*(const Vec3& v1, float t)
{
	return Vec3(
		v1.e[0] * t,
		v1.e[1] * t,
		v1.e[2] * t
	);
}

__host__ __device__ inline Vec3 operator*(float t, const Vec3& v1)
{
	return v1*t;
}

__host__ __device__ inline Vec3 operator/(const Vec3& v1, float t)
{
	return Vec3(
		v1.e[0] / t,
		v1.e[1] / t,
		v1.e[2] / t
	);
}


__host__ __device__ inline float dot(const Vec3& v1, const Vec3& v2)
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

__host__ __device__ inline Vec3& Vec3::operator*=(const float t)
{
	e[0] *= t;
	e[1] *= t;
	e[2] *= t;
	return *this;
}

__host__ __device__ inline Vec3& Vec3::operator/=(const float t)
{
	e[0] /= t;
	e[1] /= t;
	e[2] /= t;
	return *this;
}

__host__ __device__ inline bool Vec3::operator==(const Vec3& v2)
{
	return e[0] == v2.e[0] && e[1] == v2.e[1] && e[2] == v2.e[2];
}

__host__ __device__ inline Vec3 unitVector(Vec3 v)
{
	return v / v.length();
}

__host__ __device__ inline void Vec3::writeFrameBuffer(int i, int j, float* fBuffer)
{
	int index = (MAX_Y - j - 1) * MAX_X * 3 + i * 3; // Find location in frame buffer.

#ifdef ALLOWOVERFLOW
	fBuffer[index] = e[2];
	fBuffer[index + 1] = e[1];
	fBuffer[index + 2] = e[0];
#else
	fBuffer[index] = e[2] > 1 ? 1 : e[2];
	fBuffer[index + 1] = e[1] > 1 ? 1 : e[1];
	fBuffer[index + 2] = e[0] > 1 ? 1 : e[0];
#endif
}

__host__ __device__ inline void Vec3::readFrameBuffer(int i, int j, float* fBuffer)
{
	int index = (MAX_Y - j - 1) * MAX_X * 3 + i * 3; // Find location in frame buffer.

	e[2] = fBuffer[index];
	e[1] = fBuffer[index+1];
	e[0] = fBuffer[index+2];
}