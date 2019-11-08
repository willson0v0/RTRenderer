#include <iostream>
#include <opencv/cv.hpp>

int main()
{
	int x = 1000;
	int y = 500;
	cv::Mat M(y, x, CV_8UC3, cv::Scalar(0, 0, 255));
	for (int i = 0; i < y; i++)
	{
		for (int j = 0; j < x; j++)
		{
			float r = float(j) / float(x);
			float g = float(y - i) / float(y);
			float b = 0.2;

			M.at<cv::Vec3b>(i, j) = cv::Vec3b(int(255.99 * b), int(255.99 * g), int(255.99 * r));
		}
	}

	cv::namedWindow("wow", cv::WINDOW_AUTOSIZE);
	cv::imshow("wow", M);

	cv::waitKey();

	return 0;
}