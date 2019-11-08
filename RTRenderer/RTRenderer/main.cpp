#include <iostream>
#include <opencv/cv.hpp>
#include "Vec3.cpp"

int main()
{
	int x = 1000;
	int y = 500;
	cv::Mat M(y, x, CV_8UC3, cv::Scalar(0, 0, 255));
	for (int i = 0; i < y; i++)
	{
		for (int j = 0; j < x; j++)
		{
			Vec3 pix(double(j) / double(x), (double(y) - i) / double(y), 0.2);

			M.at<cv::Vec3b>(i, j) = pix.toCVPix();
		}
	}

	cv::namedWindow("wow", cv::WINDOW_AUTOSIZE);
	cv::imshow("wow", M);

	cv::waitKey();

	return 0;
}