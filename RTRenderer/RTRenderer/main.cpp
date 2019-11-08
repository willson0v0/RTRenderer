#include <iostream>
#include <opencv/cv.hpp>

int main()
{
	int x = 200;
	int y = 200;
	cv::Mat M(x, y, CV_8UC3, cv::Scalar(0, 0, 255));
	cv::MatIterator_<cv::Vec3b> it = M.begin<cv::Vec3b>();
	for (int i = y - 1; i >= 0; i--)
	{
		for (int j = 0; j < x; j++)
		{
			float r = float(i) / float(x);
			float g = float(j) / float(y);
			float b = 0.2;

			(*it)[0] = int(255.99 * r);
			(*it)[1] = int(255.99 * g);
			(*it)[2] = int(255.99 * b);
			++it;
			// if (it == M.end<cv::Vec3b>()) return 1;
		}
	}

	cv::namedWindow("wow", cv::WINDOW_AUTOSIZE);
	cv::imshow("wow", M);

	cv::waitKey();

	return 0;
}