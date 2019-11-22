#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace cv;
using std::endl;
using std::cout;

void  FindingContours() {
	Mat src = imread("lena.png");
	Mat src_gray;
	cvtColor(src, src_gray, COLOR_BGR2GRAY);
	blur(src_gray, src_gray, Size(3, 3));
	Mat canny_out;
	Canny(src_gray, canny_out, 100, 180);
	std::vector<Vec4i> hierarchy;
	std::vector<std::vector<Point>> contours;
	findContours(canny_out, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);
	Mat drawing = Mat::zeros(canny_out.size(), CV_8UC3);
	for (size_t i = 0; i < contours.size(); i++)
	{
		Scalar color = Scalar(0,0,255);
		drawContours(drawing, contours, (int)i, color, 2, LINE_8, hierarchy, 0);
	}
	imshow("Contours", drawing);
	waitKey(0);
}

