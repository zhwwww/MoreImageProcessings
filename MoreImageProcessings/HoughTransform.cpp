#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace cv;
using std::endl;
using std::cout;
using std::vector;

//Hough Line Transform is used to detect straight lines
//To apply the Transform,first an edge detection pre-processing is desirable.

void HoughLineTransform() {
	
	Mat src = imread("sudoku.png",IMREAD_GRAYSCALE);

	GaussianBlur(src, src, Size(3, 3), 0);
	medianBlur(src, src, 3);
	Mat edges;
	Canny(src, edges, 50, 200);
	//imshow("edges", edges);
	cvtColor(src, src, COLOR_GRAY2BGR);
	Mat dst = src.clone();
	Mat dstP = src.clone();

	//Standard Hough Line Transforms
	vector<Vec<float,2>> lines;
	HoughLines(edges, lines, 1, CV_PI / 180, 280, 0, 0);
	//Draw Lines
	for (size_t i = 0; i < lines.size(); i++) {
		float tho = lines[i][0], theta = lines[i][1];
		Point pt1, pt2;
		double a = cos(theta), b = sin(theta);
		double x0 = a * tho, y0 = b * tho;
		pt1.x = cvRound(x0 - 1000 * b);
		pt1.y = cvRound(y0 + 1000 * a);
		pt2.x = cvRound(x0 + 1000 * b);
		pt2.y = cvRound(y0 - 1000 * a);
		line(dst, pt1, pt2, Scalar(0, 0, 255), 1, LINE_AA);//antialiased line
	}
	imshow("HoughLines", dst);
	
	//Probabilistic Line Transform
	vector<Vec<int, 4>> linesP;
	HoughLinesP(edges, linesP, 1, CV_PI / 180, 50, 0, 10);
	for (size_t i = 0; i < linesP.size(); i++) {
		line(dstP, Point(linesP[i][0], linesP[i][1]),
			Point(linesP[i][2], linesP[i][3]), Scalar(0, 0, 255), 1, LINE_AA);
	}
	imshow("HoughLinesP", dstP);
	waitKey(0);
}

//may be unprecise especially when dealing with images which consist of a large group of circles with raidus varies rapidly
void HoughCircleTransform() {
	Mat src = imread("circle3.jpg", IMREAD_COLOR);
	Mat dst;

	Mat edges;
	cvtColor(src, dst, COLOR_BGR2GRAY);
	GaussianBlur(dst, dst, Size(3, 3), 0);
	medianBlur(dst, dst, 3);
	Canny(dst, edges, 50, 200);
	imshow("edges", edges);
	vector<Vec<float, 3>> circles;
	HoughCircles(dst, circles, HOUGH_GRADIENT, 1, src.rows / 8, 100, 100, 0, 0);

	for (size_t i = 0; i < circles.size(); i++) {
		circle(src, Point(circles[i][0], circles[i][1]), 1, Scalar(0, 255, 0), 2, LINE_AA);
		circle(src, Point(circles[i][0], circles[i][1]), circles[i][2], Scalar(255, 0, 0), 2, LINE_AA);
	}
	imshow("src", src);
	waitKey(0);
}