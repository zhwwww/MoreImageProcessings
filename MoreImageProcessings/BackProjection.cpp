#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace cv;
using std::endl;
using std::cout;

/*
What is Back Projection?
Back Projection is a way of recording how well the pixels of a given image 
fit the distribution of pixel in a histogram model.
To make it simpler, for Back Projection, you calculate the 
histogram model of a feature and then use it to find this feature
in an image.
*/
Mat hueSaturation;
Mat hueSaturation2;
void BackProjOps(int val0, int val1);
void BackProjection() {
	
	Mat src = imread("h1.jpg", IMREAD_COLOR);
	VideoCapture cap(0);
	Mat hsv;
	cvtColor(src, hsv, COLOR_BGR2HSV);
	hueSaturation.create(hsv.size(), CV_8UC2);
	int ch[] = { 0,0,1,1 };
	mixChannels(&hsv, 1, &hueSaturation, 1, ch, 2);
	namedWindow("backproj", WINDOW_AUTOSIZE);
	int hueBinsVal = 2, satBinsVal = 2;
	createTrackbar("hueBins", "backproj", &hueBinsVal, 255);
	createTrackbar("satBins", "backproj", &satBinsVal, 255);
	setTrackbarMin("hueBins", "backproj", 2);
	setTrackbarMin("satBins", "backproj", 2);
	while (true) {
		Mat frame;
		cap >> frame;
		cvtColor(frame, frame, COLOR_BGR2HSV);
		std::vector<Mat> hsv2;
		split(frame, hsv2);
		merge((Mat*)&hsv2[0], 2, hueSaturation2);	
		BackProjOps(hueBinsVal, satBinsVal);
		char c = waitKey(30);
		if (c == 'q') {
			destroyAllWindows();
			break;
		}
	}
}

void BackProjOps(int val0, int val1) {
	Mat hist;
	int histSize[2];
	histSize[0] = val0;
	histSize[1] = val1;
	float hue_range[] = { 0.f,255.f };
	float sat_range[] = { 0.f,255.f };
	const float* ranges[] = { hue_range, sat_range };
	int channels[] = { 0,1 };

	calcHist(&hueSaturation, 1, channels, Mat(), hist, 2, histSize, ranges, true, false);
	normalize(hist, hist, 0, 255, NORM_MINMAX, -1, Mat());
	Mat backproj;
	calcBackProject(&hueSaturation2, 1, channels, hist, backproj, ranges, 1, true);
	//imshow("src", src);
	imshow("backproj", backproj);
}