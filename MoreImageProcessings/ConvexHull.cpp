#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace cv;
using std::endl;
using std::cout;

void ConvexHull() {
	Mat src = imread("house.png", IMREAD_COLOR);
	Mat src_gray;
	cvtColor(src, src_gray, COLOR_BGR2GRAY);
	Canny(src_gray, src_gray, 100, 200);
	std::vector<std::vector<Point>> contours;
	findContours(src_gray, contours, RETR_TREE, CHAIN_APPROX_SIMPLE);
	std::vector<std::vector<Point>> hull(contours.size());
	for (size_t i = 0; i < contours.size(); i++) {
		convexHull(contours[i], hull[i]);
	}
	Mat drawing = Mat::zeros(src_gray.size(), CV_8UC3);
	for (size_t i = 0; i < contours.size(); i++)
	{
		Scalar color = Scalar(0, 0, 255);
		drawContours(drawing, contours, (int)i, color);
		drawContours(drawing, hull, (int)i, color);
	}
	imshow("Contours", drawing);
	waitKey(0);
}
