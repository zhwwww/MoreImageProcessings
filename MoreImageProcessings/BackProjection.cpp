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
Mat hsv;
Mat hsv2;
Mat mask;
Mat src;
Mat frame;
int up = 30, low = 30;


void BackProjOps(int val0, int val1);
void PickPoint(int event, int x, int y, int, void*);
void BackProjection() {
	VideoCapture cap(0);
	cap >> src;
	cvtColor(src, hsv, COLOR_BGR2HSV);
	namedWindow("src", WINDOW_AUTOSIZE);
	imshow("src", src);
	setMouseCallback("src", PickPoint, 0);
	waitKey(0);
	destroyWindow("src");
	destroyWindow("mask");
	int hueBinsVal = 2, satBinsVal = 2;
	namedWindow("show", WINDOW_AUTOSIZE);
	createTrackbar("hueBins", "show", &hueBinsVal, 255);
	createTrackbar("satBins", "show", &satBinsVal, 255);
	setTrackbarMin("hueBins", "show", 2);
	setTrackbarMin("satBins", "show", 2);
	while (true) {
		cap >> frame;
		cvtColor(frame, hsv2, COLOR_BGR2HSV);	
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
	/*
	  2D Hue-Saturation Histogram
	*/
	calcHist(&hsv, 1, channels, mask, hist, 2, histSize, ranges, true, false);
	normalize(hist, hist, 0, 255, NORM_MINMAX, -1, Mat());
	Mat backproj;
	calcBackProject(&hsv2, 1, channels, hist, backproj, ranges, 1, true);
	/*int w = 400, h = 400;
	Mat histoGram = Mat::zeros(h,w,CV_8UC3);
	int bin_w = cvRound((double)w / histSize[1]);*/
	//normalize(hist, hist, 0, maxHist, NORM_MINMAX, -1);
	//hist.convertTo(hist, -1, 255.0 / sum(hist)[0], 0);
	//for (int i = 0; i < histSize[1] - 1; i++) {
	//	for()
	//	/*rectangle(histoGram, Point(i * bin_w, h),
	//		Point((i + 1) * bin_w, h - cvRound(hist.at<float>(i) * h / 255.0f) ),
	//		Scalar(0,0,255),FILLED);	*/
	//	line(histoGram, Point(i * bin_w, h - cvRound(hist.at<float>(0,i) * h)),
	//		Point((i + 1) * bin_w, h - cvRound(hist.at<float>(0,i) * h)), Scalar(0, 0, 255), 1, 8);
	//}
	imshow("histogram", hist);
	Mat kernel = getStructuringElement(MORPH_CROSS, Size(3, 3));
	erode(backproj, backproj, kernel);
	dilate(backproj, backproj, kernel);
	medianBlur(backproj, backproj, 3);
	Mat show;
	frame.copyTo(show, backproj);
	imshow("show", show);

}

void PickPoint(int event, int x, int y, int , void* ) {

	if (event != EVENT_LBUTTONDOWN)
	{
		return;
	}
	Mat src2;
	src2 = src.clone();
	Point seed = Point(x, y);

	int newMaskVal = 255;
	Scalar newVal = Scalar(0, 0, 255);

	int connectivity = 8;
	int flags = connectivity + (newMaskVal << 8) + FLOODFILL_FIXED_RANGE;

	Mat mask2 = Mat::zeros(src.rows + 2, src.cols + 2, CV_8U);
	floodFill(src2, mask2, seed, newVal, 0, Scalar(low, low, low), Scalar(up, up, up), flags);
	mask = mask2(Range(1, mask2.rows - 1), Range(1, mask2.cols - 1));
	imshow("src", src2);
	imshow("mask", mask);
}
