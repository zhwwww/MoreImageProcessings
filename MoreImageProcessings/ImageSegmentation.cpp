#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace cv;
using std::endl;
using std::cout;
using std::vector;
void ImageSegmentation() {

	Mat src = imread("cards.png", IMREAD_COLOR);    
	//change white backgounds to black backgounds
	//medianBlur(src, src, 3);
	GaussianBlur(src, src, Size(3, 3), 0);
	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {
			Vec3b v1 = src.at<Vec3b>(i, j);
			if (v1[0] > 250 && v1[1] > 250 && v1[2] > 250)
			{
				src.at<Vec3b>(i, j)[0] = 0;
				src.at<Vec3b>(i, j)[1] = 0;
				src.at<Vec3b>(i, j)[2] = 0;
			}
		}
	}
	imshow("src", src);
	//imshow("Black Background Image", src);
	Mat kernel = (Mat_<float>(3, 3) <<
		1, 1, 1,
		1, -8, 1,
		1, 1, 1);
	Mat imgLaplacian;
	filter2D(src, imgLaplacian, CV_32FC3, kernel);
	Mat sharp;
	src.convertTo(sharp, CV_32FC3);
	Mat imgResult = sharp - 2*imgLaplacian;
	imgResult.convertTo(imgResult, CV_8UC3);
	imgLaplacian.convertTo(imgLaplacian, CV_8UC3);
	//imshow("Laplace Filtered Image", imgLaplacian);
	//imshow("New Sharped Image", imgResult);
	Mat bw;
	cvtColor(imgResult, bw, COLOR_BGR2GRAY);
	threshold(bw, bw, 180, 255, THRESH_BINARY);
	//imshow("bw", bw); 	
	Mat dist;
	distanceTransform(bw, dist, DIST_L2, 3);
	normalize(dist, dist, 0, 1.0, NORM_MINMAX);
	//imshow("Distance Transform Image", dist);
	threshold(dist, dist, 0.4, 1.0, THRESH_BINARY);
	//imshow("Binary Distance Transform Image", dist);
	/*Mat kernel3 = Mat::ones(3, 3, CV_8U);
	dilate(dist, dist, kernel3);
	imshow("Peaks", dist);*/
	dist.convertTo(dist, CV_8U);
	vector<vector<Point>> contours;
	Mat markers = Mat::zeros(dist.size(), CV_32S);
	//mark unknown regions as 0
	findContours(dist, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
	for (size_t i = 0; i < contours.size(); i++){
		drawContours(markers, contours, (int)i, Scalar(1 + (int)i), -1);
	}
	circle(markers, Point(5, 5), 3, Scalar(100), -1);
	watershed(imgResult, markers);
	Mat dst = Mat::zeros(markers.size(), CV_8UC3);
	vector<Vec3b> colors;
	for (size_t i = 0; i < contours.size(); i++)
	{
		int b = theRNG().uniform(0, 256);
		int g = theRNG().uniform(0, 256);
		int r = theRNG().uniform(0, 256);
		colors.push_back(Vec3b((uchar)b, (uchar)g, (uchar)r));
	}
	for (int i = 0; i < markers.rows; i++)
	{
		for (int j = 0; j < markers.cols; j++)
		{
			int index = markers.at<int>(i, j);
			if (index > 0 && index <= static_cast<int>(contours.size()))
			{
				dst.at<Vec3b>(i, j) = colors[index - 1];
			}
		}
	}
	imshow("Final Result", dst);
	waitKey(0);
}
