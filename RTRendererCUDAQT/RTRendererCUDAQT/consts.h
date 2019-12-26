#pragma once
#include <ctime>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <curand_kernel.h>

#define ALLOWOUTOFBOUND
#define ALLOWOVERFLOW
//#define USE_BVH
//#define DARKSCENE

enum class LogLevel
{
	fatal	= 0x00,
	error	= 0x01,
	warning = 0x02,
	info	= 0x03,
	debug	= 0x04,
	extra	= 0x05
};

constexpr auto MAX_X = 1024;
constexpr auto MAX_Y = 768;
//constexpr auto MAX_X = 512;
//constexpr auto MAX_Y = 384;
constexpr auto BLK_X = 25;
constexpr auto BLK_Y = 20;
constexpr auto PI = 3.1415926535897932384626433832795;
constexpr auto logLevel = LogLevel::extra;
constexpr auto ITER = 7;
constexpr auto SPP = 4;


constexpr auto CAMERA_NUM = 12;
constexpr auto WORLD_NUM = 1;
constexpr auto RENDER_NUM = 2;
constexpr auto OBJECT_NUM = 1;


constexpr auto ANSI_COLOR_RED = "\x1b[31m";
constexpr auto ANSI_COLOR_GREEN = "\x1b[32m";
constexpr auto ANSI_COLOR_YELLOW = "\x1b[33m";
constexpr auto ANSI_COLOR_BLUE = "\x1b[34m";
constexpr auto ANSI_COLOR_MAGENTA = "\x1b[35m";
constexpr auto ANSI_COLOR_CYAN = "\x1b[36m";
constexpr auto ANSI_COLOR_RESET = "\x1b[0m";

__device__ __managed__ bool VTModeEnabled = false;
