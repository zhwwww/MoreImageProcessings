#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace cv;
using std::endl;
using std::cout;
using std::vector;
using std::complex;
static void createPSF(Mat& h_PSF, Size size_src, int R);
static void shiftFFT(Mat& im1, Mat& im2);
static void freq2D(Mat& src, Mat& Hw, Mat* out);
static void calcHw(Mat& Hw, Mat& h_PSF, int snr);
static void RCallback(int val, void*);
static void Mat2vector(Mat& src, vector<complex<float>>& dst);
static void Vector2Mat(vector<complex<float>>& src, int rows, int cols, Mat& dst);
static int snr = 2000;
static Mat src2;
void Out_of_focusDeblurFilter() {
	
	Mat src = imread("original.jpg", IMREAD_GRAYSCALE);
	src2 = Mat(src, Rect(0, 0, src.cols & -2, src.rows & -2));
	GaussianBlur(src2, src2, Size(3, 3), 0);
	normalize(src2, src2, 0, 1, NORM_MINMAX, CV_32F);
	imshow("src", src2);
	namedWindow("out1", WINDOW_AUTOSIZE);
	createTrackbar("R", "out1", NULL, 40, RCallback);
	RCallback(0, 0);
	waitKey(0);
}

static void RCallback(int val, void*) {
	Mat h_PSF;
	createPSF(h_PSF, src2.size(), val);
	Mat Hw;
	calcHw(Hw, h_PSF, snr);
	Mat out[2];
	freq2D(src2, Hw, out);
	imshow("out1", out[0]);
//	imshow("out2", out[1]);
}
//point spread function
static void createPSF(Mat& h_PSF, Size size_src,int R) {
	h_PSF.create(size_src, CV_32F);
	h_PSF = Scalar(0);
	circle(h_PSF, Point(size_src.width / 2, size_src.height / 2), R, Scalar(100), -1, 8);
	Scalar summ = sum(h_PSF);
	h_PSF /= summ[0];
}

static void calcHw(Mat& Hw, Mat& h_PSF, int snr) {
	shiftFFT(h_PSF, h_PSF);
	Mat planes[2] = { Mat_<float>(h_PSF.clone()), Mat::zeros(h_PSF.size(), CV_32F) };
	Mat complexI;
	merge(planes, 2, complexI);
	dft(complexI, complexI);
	vector<complex<float>> vComplex((size_t)complexI.cols*complexI.rows);
	Mat2vector(complexI, vComplex);
	for (size_t i = 0; i < vComplex.size(); i++) {
		//in case of becoming nan by diving 0 , add 1e-7
		vComplex[i] = (vComplex[i] * vComplex[i]) / ( vComplex[i] + complex<float>(1e-7, 1e-7)) / (vComplex[i] * vComplex[i] + complex<float>(1 / (float)snr, 0));
	}
	Vector2Mat(vComplex, complexI.rows, complexI.cols, Hw);
	/*split(complexI, planes);
	planes[1] = -planes[1];
	Mat complexI2;
	merge(planes, 2, complexI2);
	Mat squareComplexI;
	mulSpectrums(complexI, complexI2, squareComplexI, 0);
	split(squareComplexI, planes);
	planes[0] = planes[0] / (planes[0] + 1 / (float)snr);
	merge(planes, 2, squareComplexI);
	mulSpectrums(1 / complexI, squareComplexI, complexI, 0);
	Hw = complexI;*/
}

static void freq2D(Mat& src, Mat& Hw, Mat* out) {

	out[0] = Mat_<float>(src.clone());
	out[1] = Mat_<float>(src.size(), 0);
	Mat complexI;
	merge(out, 2, complexI);
	dft(complexI, complexI);
	
	//mulSpectrums(complexI, Hw, complexI, 0);
	vector<complex<float>> vComplex((size_t)complexI.cols * complexI.rows);
	Mat2vector(complexI, vComplex);
	vector<complex<float>> HwComplex((size_t)Hw.cols * Hw.rows);
	Mat2vector(Hw, HwComplex);
	for (size_t i = 0; i < vComplex.size(); i++) {
		vComplex[i] = (vComplex[i] * HwComplex[i]);
	}
	Vector2Mat(vComplex, complexI.rows, complexI.cols, complexI);
	idft(complexI, complexI);
	split(complexI, out);
	normalize(out[0], out[0], 0, 255, NORM_MINMAX, CV_8U);
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

static void Mat2vector(Mat& src, vector<complex<float>>& dst)
{
	MatIterator_<float> it1, it2;
	Mat planes[2];
	split(src, planes);
	it1 = planes[0].begin<float>();
	it2 = planes[1].begin<float>();
	for (size_t i = 0; i < dst.size(); i++)
	{
		dst[i].real(*it1++);
		dst[i].imag(*it2++);
	}
}

static void Vector2Mat(vector<complex<float>>& src, int rows,int cols,Mat& dst)
{
	MatIterator_<float> it1, it2;
	Mat planes[2] = { Mat_<float>(rows,cols), Mat_<float>(rows,cols) };
	it1 = planes[0].begin<float>();
	it2 = planes[1].begin<float>();
	for (size_t i = 0; i < src.size(); i++) {
		*it1++ = src[i].real();
		*it2++ = src[i].imag();
	}
	merge(planes, 2, dst);
}
