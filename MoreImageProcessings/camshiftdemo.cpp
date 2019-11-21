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
static Mat hsv;
static Mat hsv2;
static Mat src;
static Mat frame;
static Rect roi_rect;

static void BackProjOps(int val0, int val1);
static void PickPoint(int event, int x, int y, int, void*);
void camshiftdemo() {
	VideoCapture cap(0);
	cap >> src;
	cvtColor(src, hsv, COLOR_BGR2HSV);
	namedWindow("src", WINDOW_AUTOSIZE);
	imshow("src", src);
	setMouseCallback("src", PickPoint, 0);
	waitKey(0);
	destroyWindow("src");
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

static void BackProjOps(int val0, int val1) {
	Mat hist;
	int histSize[2];
	histSize[0] = val0;
	histSize[1] = val1;
	float hue_range[] = { 0.f,255.f };
	float sat_range[] = { 0.f,255.f };
	const float* ranges[] = { hue_range, sat_range };
	int channels[] = { 0,1 };
	Mat hsv_roi(hsv,roi_rect);
	calcHist(&hsv_roi, 1, channels, Mat(), hist, 2, histSize, ranges, true, false);
	normalize(hist, hist, 0, 255, NORM_MINMAX, -1, Mat());
	Mat backproj;
	calcBackProject(&hsv2, 1, channels, hist, backproj, ranges, 1, true);
	/*int w = 400, h = 400;
	Mat histoGram = Mat::zeros(h, w, CV_8UC3);
	int bin_w = cvRound((double)w / histSize[0]);*/
	//normalize(hist, hist, 0, maxHist, NORM_MINMAX, -1);
	//hist.convertTo(hist, -1, 1.0 / sum(hist)[0], 0);
	//for (int i = 0; i < histSize[0] - 1; i++) {
	//	/*rectangle(histoGram, Point(i * bin_w, h),
	//		Point((i + 1) * bin_w, h - cvRound(hist.at<float>(i) * h / 255.0f) ),
	//		Scalar(0,0,255),FILLED);*/
	//	line(histoGram, Point(i * bin_w, h - cvRound(hist.at<float>(i) * h)),
	//		Point((i + 1) * bin_w, h - cvRound(hist.at<float>(i + 1) * h)), Scalar(0, 0, 255), 1, 8);
	//}
	//imshow("histogram", histoGram);
	Mat kernel = getStructuringElement(MORPH_CROSS, Size(3, 3));
	erode(backproj, backproj, kernel);
	dilate(backproj, backproj, kernel);
	medianBlur(backproj, backproj, 3);
	Mat show;
	frame.copyTo(show, backproj);
	imshow("show", show);

}

static void PickPoint(int event, int x, int y, int, void*) {
	static Point origin, last;
	static bool buttonDown = false;
	if (event == EVENT_LBUTTONDOWN) {
		if (buttonDown == false) {
			origin.x = x;
			origin.y = y;
			buttonDown = true;
		}
	}
	if (buttonDown == true) {
		last.x = x;
		last.y = y;
	}
	if (event == EVENT_LBUTTONUP && buttonDown == true) {
		buttonDown = false;
		roi_rect.x = origin.x >= last.x ? last.x : origin.x;
		roi_rect.y = origin.y >= last.y ? last.y : origin.y;
		roi_rect.width = origin.x >= last.x ? (origin.x - last.x) : (last.x - origin.x);
		roi_rect.height = origin.y >= last.y ? (origin.y - last.y) : (last.y - origin.y);
		/*cout << roi_rect.x << endl;
		cout << roi_rect.y << endl;
		cout << roi_rect.width << endl;
		cout << roi_rect.height << endl;*/
		Mat src2;
		src2 = src.clone();
		rectangle(src2, roi_rect, Scalar(0, 0, 255), 2);
		imshow("src", src2);
	}
}