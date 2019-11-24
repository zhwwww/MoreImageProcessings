#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace cv;
using std::endl;
using std::cout;
using std::vector;
using std::complex;

static void createPSF(Mat& h_PSF, Size size_src, int len, float theta);
static void shiftFFT(Mat& im1, Mat& im2);
static void freq2D(Mat& src, Mat& Hw, Mat* out);
static void calcHw(Mat& Hw, Mat& h_PSF, int snr);
static void RCallback(int val, void*);
static void Mat2vector(Mat& src, vector<complex<float>>& dst);
static void Vector2Mat(vector<complex<float>>& src, int rows, int cols, Mat& dst);
void edgetaper(const Mat& inputImg, Mat& outputImg, double gamma, double beta);
static int snr = 100;
static Mat src2;
void MotionDeblurFilter() {

	Mat src = imread("deblur4.jpg", IMREAD_GRAYSCALE);
	src2 = Mat(src, Rect(0, 0, src.cols & -2, src.rows & -2));
//	GaussianBlur(src2, src2, Size(3, 3), 0);
//normalize(src2, src2, 0, 1, NORM_MINMAX, CV_32F);
	imshow("src", src2);
	namedWindow("out1", WINDOW_AUTOSIZE);
	createTrackbar("LEN", "out1", NULL, 200, RCallback,(void*)uchar(0));
	createTrackbar("THETA", "out1", NULL, 90, RCallback,(void*)uchar(1));
	RCallback(0, (void*)uchar(1));
	waitKey(0);
	destroyAllWindows();
}

static void RCallback(int val, void* user_data) {
	static int theta = 0;
	static int len = 1;
	uchar i = (uchar)user_data;
	if (i == 0) {
		len = val;
	}
	else {
		theta = val;
	}
	Mat h_PSF;
	createPSF(h_PSF, src2.size(), len, theta);
	//imshow("psf", h_PSF);
	Mat Hw;
	calcHw(Hw, h_PSF, snr);
	src2.convertTo(src2, CV_32F);
	//edgetaper(src2, src2, 5.0, 0.2);
	Mat out[2];
	freq2D(src2, Hw, out);
	imshow("out1", out[0]);
	//	imshow("out2", out[1]);
}
//point spread function
static void createPSF(Mat& h_PSF, Size size_src, int len, float theta) {
	h_PSF.create(size_src, CV_32F);
	h_PSF = Scalar(0);
	Point point(size_src.width / 2, size_src.height / 2);
	ellipse(h_PSF, point, Size(0, cvRound(float(len) / 2.0)), 90.0 - theta, 0, 360, Scalar(100), -1);
	Scalar summ = sum(h_PSF);
	h_PSF /= summ[0];
}

static void calcHw(Mat& Hw, Mat& h_PSF, int snr) {
	shiftFFT(h_PSF, h_PSF);
	Mat planes[2] = { Mat_<float>(h_PSF.clone()), Mat::zeros(h_PSF.size(), CV_32F) };
	Mat complexI;
	merge(planes, 2, complexI);
	dft(complexI, complexI);
	vector<complex<float>> vComplex((size_t)complexI.cols * complexI.rows);
	Mat2vector(complexI, vComplex);
	for (size_t i = 0; i < vComplex.size(); i++) {
		//in case of becoming nan by diving 0 , add 1e-7
		vComplex[i] = (vComplex[i] * vComplex[i]) / (vComplex[i] + complex<float>(1e-7, 1e-7)) / (vComplex[i] * vComplex[i] + complex<float>(1 / (float)snr, 0));
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

static void Vector2Mat(vector<complex<float>>& src, int rows, int cols, Mat& dst)
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

void edgetaper(const Mat& inputImg, Mat& outputImg, double gamma, double beta)
{
	int Nx = inputImg.cols;
	int Ny = inputImg.rows;
	Mat w1(1, Nx, CV_32F, Scalar(0));
	Mat w2(Ny, 1, CV_32F, Scalar(0));
	float* p1 = w1.ptr<float>(0);
	float* p2 = w2.ptr<float>(0);
	float dx = float(2.0 * CV_PI / Nx);
	float x = float(-CV_PI);
	for (int i = 0; i < Nx; i++)
	{
		p1[i] = float(0.5 * (tanh((x + gamma / 2) / beta) - tanh((x - gamma / 2) / beta)));
		x += dx;
	}
	float dy = float(2.0 * CV_PI / Ny);
	float y = float(-CV_PI);
	for (int i = 0; i < Ny; i++)
	{
		p2[i] = float(0.5 * (tanh((y + gamma / 2) / beta) - tanh((y - gamma / 2) / beta)));
		y += dy;
	}
	Mat w = w2 * w1;
	multiply(inputImg, w, outputImg);
}