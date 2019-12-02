#pragma once
#include <ctime>

#define ALLOWOUTOFBOUND
#define ALLOWOVERFLOW
#define DARKSCENE
//#define USE_BVH

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
constexpr auto BLK_X = 25;
constexpr auto BLK_Y = 20;
constexpr auto PI = 3.1415926535897932384626433832795;
constexpr auto logLevel = LogLevel::debug;

constexpr auto ANSI_COLOR_RED = "\x1b[31m";
constexpr auto ANSI_COLOR_GREEN = "\x1b[32m";
constexpr auto ANSI_COLOR_YELLOW = "\x1b[33m";
constexpr auto ANSI_COLOR_BLUE = "\x1b[34m";
constexpr auto ANSI_COLOR_MAGENTA = "\x1b[35m";
constexpr auto ANSI_COLOR_CYAN = "\x1b[36m";
constexpr auto ANSI_COLOR_RESET = "\x1b[0m";

__device__ __managed__ bool VTModeEnabled = false;
__device__ __managed__ clock_t StartTime;