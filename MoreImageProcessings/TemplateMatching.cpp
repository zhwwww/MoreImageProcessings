#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace cv;
using std::endl;
using std::cout;

// Template matching 
// Finding areas of an image that match (are similar) to a template image.

static Mat src;
static Rect roi_rect;
static void PickPoint(int event, int x, int y, int, void*);
void TemplateMatching() {
	VideoCapture cap(0);
	cap >> src;
	namedWindow("src", WINDOW_AUTOSIZE);
	imshow("src", src);
	setMouseCallback("src", PickPoint, 0);
	waitKey(0);
	destroyWindow("src");
	int method=0;
	namedWindow("match", WINDOW_AUTOSIZE);
	createTrackbar("Method", "match", &method, 5);
	Mat templ(src, roi_rect);
	while (true) {
		Mat frame;
		cap >> frame;
		Mat result;
		result.create(frame.rows - templ.rows + 1, frame.cols - templ.cols + 1,CV_32F);
		matchTemplate(frame, templ, result, method);
		normalize(result, result, 0, 255, NORM_MINMAX, CV_8U);
		Point minLoc, maxLoc;
		minMaxLoc(result, NULL, NULL, &minLoc, &maxLoc);
		Point matchLoc;
		if (method == TM_SQDIFF || method == TM_SQDIFF_NORMED)
		{
			matchLoc = minLoc;
		}
		else
		{
			matchLoc = maxLoc;
		}
		rectangle(frame, Rect(matchLoc, templ.size()), Scalar(0, 0, 255), 2, 8);
		imshow("match", frame);
		char c = waitKey(30);
		if (c == 'q') {
			break;
		}
	}

}

static void PickPoint(int event, int x, int y, int, void*) {
	static Point origin, last;
	static bool buttonDown = false;
	Mat src2;
	Rect rect;
	src2 = src.clone();
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
		rect.x = origin.x >= last.x ? last.x : origin.x;
		rect.y = origin.y >= last.y ? last.y : origin.y;
		rect.width = origin.x >= last.x ? (origin.x - last.x) : (last.x - origin.x);
		rect.height = origin.y >= last.y ? (origin.y - last.y) : (last.y - origin.y);
		rectangle(src2, rect, Scalar(0, 255, 0), 2);
		imshow("src", src2);
	}
	if (event == EVENT_LBUTTONUP && buttonDown == true) {
		buttonDown = false;
		roi_rect = rect;
	}
}
