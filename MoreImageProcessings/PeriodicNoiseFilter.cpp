#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace cv;
using std::endl;
using std::cout;
static void shiftFFT(Mat& im1, Mat& im2);
void filter2DFreq(Mat& img, Mat& H, Mat& out);
void MouseCallBack(int event, int x, int y, int flags, void* userdata);
Mat PSD;
Mat H;
int radius = 21;
//remove periodic noise in the Fourier domain
void calcPSD(Mat& imgIn, Mat& PSD, bool flag);
void PeriodNoiseFilter() {
	Mat src = imread("period_input.jpg", IMREAD_GRAYSCALE);
	Rect rect_roi(0, 0, src.cols & -2, src.rows & -2);
	src = src(rect_roi);
	calcPSD(src, PSD, true);
	H = Mat(src.size(), CV_32F, Scalar(1));
	imshow("src", src);
	imshow("PSD", PSD);
	setMouseCallback("PSD", MouseCallBack);
	waitKey(0);
	shiftFFT(H, H);
	Mat out;
	filter2DFreq(src, H, out);
	normalize(out, out, 0, 255, NORM_MINMAX, CV_8U);
	imshow("out", out);
	waitKey(0);
}

void MouseCallBack(int event, int x, int y, int flags, void* userdata) {
	if (event == EVENT_LBUTTONDOWN) {
		Point center(x, y);
		Point c2 = center, c3 = center, c4 = center;
		c2.y = PSD.rows - center.y;
		c3.x = PSD.cols - center.x;
		c4 = Point(c3.x, c2.y);
		circle(PSD, center, radius, 0, -1, 8);
		circle(PSD, c2, radius, 0, -1, 8);
		circle(PSD, c3, radius, 0, -1, 8);
		circle(PSD, c4, radius, 0, -1, 8);
		circle(H, center, radius, 0, -1, 8);
		circle(H, c2, radius, 0, -1, 8);
		circle(H, c3, radius, 0, -1, 8);
		circle(H, c4, radius, 0, -1, 8);
		imshow("PSD", PSD);
	}
	
}
void calcPSD(Mat& imgIn, Mat& PSD, bool flag) {
	Mat planes[2] = { Mat_<float>(imgIn.clone()),Mat::zeros(imgIn.size(),CV_32F) };
	Mat complexI;
	merge(planes, 2, complexI);
	dft(complexI, complexI);
	shiftFFT(complexI, complexI);
	split(complexI, planes);
	//remove dc 
	planes[0].at<float>(0) = 0;
	planes[1].at<float>(0) = 0;
	Mat mag;
	magnitude(planes[0], planes[1], mag);
	Mat psd;
	multiply(mag, mag, psd);
	if (flag) {
		psd += Scalar(1);
		log(psd, psd);
	}
	normalize(psd, psd, 0, 255, NORM_MINMAX ,CV_8U);
	PSD = psd;
}

static void shiftFFT(Mat& im1, Mat& im2) {
	int a = im1.cols / 2;
	int b = im1.rows / 2;
	im2 = im1.clone();
	Mat q0 = Mat(im2, Rect(0, 0, a, b));
	Mat q1 = Mat(im2, Rect(a, 0, a, b));
	Mat q2 = Mat(im2, Rect(0, b, a, b));
	Mat q3 = Mat(im2, Rect(a, b, a, b));
	Mat tmp;
	q0.copyTo(tmp);
	q3.copyTo(q0);
	tmp.copyTo(q3);
	q1.copyTo(tmp);
	q2.copyTo(q1);
	tmp.copyTo(q2);
}

void filter2DFreq(Mat& img, Mat& H, Mat& out) {
	Mat planes[2] = { Mat_<float>(img.clone()),Mat::zeros(img.size(),CV_32F) };
	Mat complexI;
	merge(planes, 2, complexI);
	dft(complexI, complexI);
	Mat planesH[2] = { Mat_<float>(H.clone()),Mat::zeros(H.size(),CV_32F) };
	Mat complexH;
	merge(planesH, 2, complexH);
	mulSpectrums(complexI, complexH, complexI, 0);
	idft(complexI, complexI);
	split(complexI, planes);
	out = planes[0];
}