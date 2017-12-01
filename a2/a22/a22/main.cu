
/***********************************************************
* Color Transfer Implementation
************************************************************
* This code is an implementation of the paper [Reinhard2001].
* The program transfers the color of one image (in this code
* reference image) to another image (in this code target image).
*
* usage: > ColorTransfer.exe [target image] [reference image]
*
* This code is this programmed by 'tatsy'. You can use this
* code for any purpose :-)
************************************************************/

#pragma warning( disable : 4996)
#include <iostream>
#include <iomanip>
#include <cstdio>
#include <cstdlib>
using namespace std;

#include <opencv2\opencv.hpp>

//CUDA
#include "cuda_runtime.h"
#include <cublas_v2.h>
#include <device_launch_parameters.h>
#include "Color3d.h"

//A2 START
#define BLOCK_SIZE 256

__device__ void matvec(float* d_A, float* d_B, float* d_C)
{

	/*
	float sum = 0;

	for (int i = 0; i < 3; ++i) {
		sum += d_A[i] * d_B[(i * n) + tid];
	}

	d_C[0] = sum;
	*/


		int tid = threadIdx.x + blockIdx.x * blockDim.x;

		float sum = 0;
		if (tid < 3)
		{
			for (int i = 0; i < 3; ++i)
			{
				sum += d_A[i] * d_B[(i * 3) + tid];
			}

			d_C[tid] = sum;
		}


}

__global__ void matvec_kernel(float* d_A, float* d_RGB2, float* d_LMS2, float* d_C,
	const int n, int targetrows, int targetcols, float* d_Tar)
{
	const double eps = 1.0e-4;
	for (int y = 0; y < targetrows; ++y) {
		for (int x = 0; x < targetcols; ++x) {
			memcpy(&d_A, &d_Tar[y * 3 + x], 3 * sizeof(float));

			matvec(d_A, d_RGB2, d_C);
			memcpy(&d_A, d_C, 3 * sizeof(float));

			for (int c = 0; c < 3; c++)
				d_A[c] = d_A[c] > -5.0 ? pow((double)10.0, (double)d_A[c]) : eps;

			matvec(d_A, d_LMS2, d_C);
			memcpy(&d_Tar[y * 3 + x], d_C, 3 * sizeof(float));
		}
	}
}

inline void vecTransfer(float* h, Color3d* v)
{
	for (int j = 0; j < 3; ++j)
		h[j] = v->v[j];
}

//KERNEL Helper function does setup and launch
void matvec_L(cv::Mat* mRGB2LMS, cv::Mat* mLMS2lab, float* h_C, int tarrow, int tarcol, float* h_Tar)
{
	float *h_A, *h_RGB2, *h_LMS2, *d_Tar;
	float *d_A, *d_RGB2, *d_LMS2, *d_C;

	int N = 3;

	h_A = (float*)malloc(sizeof(float) * N);
	h_RGB2 = new float[mRGB2LMS->total()];
	h_LMS2 = new float[mLMS2lab->total()];
	//h_C = (float*)malloc(sizeof(float) * N);

	cudaMalloc((void**)&d_A, sizeof(float) * N);
	cudaMalloc((void**)&d_RGB2, sizeof(float) * N * N);
	cudaMalloc((void**)&d_LMS2, sizeof(float) * N * N);
	cudaMalloc((void**)&d_C, sizeof(h_C));
	cudaMalloc((void**)&d_Tar, sizeof(h_Tar));

	Color3d vec;

	//copy vec and matrix to host pointers
	vecTransfer(h_A, &vec);
	memcpy(h_RGB2, mRGB2LMS->data, mRGB2LMS->total());
	memcpy(h_LMS2, mLMS2lab->data, mLMS2lab->total());

	cudaMemcpy(d_A, h_A, sizeof(float) * N, cudaMemcpyHostToDevice);
	cudaMemcpy(d_RGB2, h_RGB2, sizeof(float) * N * N, cudaMemcpyHostToDevice);
	cudaMemcpy(d_LMS2, h_LMS2, sizeof(float) * N * N, cudaMemcpyHostToDevice);
	cudaMemcpy(d_Tar, h_Tar, sizeof(h_Tar), cudaMemcpyHostToDevice);

	matvec_kernel << <N / BLOCK_SIZE + 1, BLOCK_SIZE >> >(d_A, d_RGB2, d_LMS2, d_C, N, tarrow, tarcol, d_Tar);
	//printf("error code: %s\n",cudaGetErrorString(cudaGetLastError()));

	cudaMemcpy(h_C, d_C, sizeof(h_C), cudaMemcpyDeviceToHost);

	free(h_A);
	free(h_RGB2);
	free(h_LMS2);
	//free(h_C);

	cudaFree(d_A);
	cudaFree(d_RGB2);
	cudaFree(d_LMS2);
	cudaFree(d_C);
	cudaFree(d_Tar);
}
// End of A2 added functions

// NOTE(marko) : most instances have been replaced but leave just incase
// Multiplication of matrix and vector
Color3d operator *(const cv::Mat& M, Color3d& v) {
	Color3d u = Color3d();
	for (int i = 0; i < 3; i++) {
		u(i) = 0.0;
		for (int j = 0; j < 3; j++) {
			u(i) += M.at<double>(i, j) * v(j);
		}
	}
	return u;
}

// Transformation from RGB to LMS
const double RGB2LMS[3][3] = {
	{ 0.3811, 0.5783, 0.0402 },
	{ 0.1967, 0.7244, 0.0782 },
	{ 0.0241, 0.1288, 0.8444 }
};

// Transformation from LMS to RGB
const double LMS2RGB[3][3] = {
	{ 4.4679, -3.5873,  0.1193 },
	{ -1.2186,  2.3809, -0.1624 },
	{ 0.0497, -0.2439,  1.2045 }
};

// First transformation from LMS to lab
const double LMS2lab1[3][3] = {
	{ 1.0 / sqrt(3.0), 0.0, 0.0 },
	{ 0.0, 1.0 / sqrt(6.0), 0.0 },
	{ 0.0, 0.0, 1.0 / sqrt(2.0) }
};

// Second transformation from LMS to lab
const double LMS2lab2[3][3] = {
	{ 1.0,  1.0,  1.0 },
	{ 1.0,  1.0, -2.0 },
	{ 1.0, -1.0,  0.0 }
};

const double eps = 1.0e-4;

int main(int argc, char** argv) {
	// Check number of arguments
	if (argc <= 2) {
		cout << "usage: > ColorTransfer.exe [target image] [reference image]" << endl;
		return -1;
	}



	// Load target image
	cv::Mat target = cv::imread(argv[1], CV_LOAD_IMAGE_COLOR);
	if (target.empty()) {
		cout << "Failed to load file \"" << argv[1] << "\"" << endl;
		return -1;
	}
	cv::cvtColor(target, target, CV_BGR2RGB);
	target.convertTo(target, CV_64FC3, 1.0 / 255.0);

	// Load reference image
	cv::Mat refer = cv::imread(argv[2], CV_LOAD_IMAGE_COLOR);
	if (refer.empty()) {
		cout << "Failed to load file \"" << argv[2] << "\"" << endl;
		return -1;
	}
	cv::cvtColor(refer, refer, CV_BGR2RGB);
	refer.convertTo(refer, CV_64FC3, 1.0 / 255.0);

	// Construct transformation matrix
	const size_t bufsize = sizeof(double) * 3 * 3;
	cv::Mat mRGB2LMS = cv::Mat(3, 3, CV_64FC1);
	memcpy(mRGB2LMS.data, &RGB2LMS[0][0], bufsize);

	cv::Mat mLMS2RGB = cv::Mat(3, 3, CV_64FC1);
	memcpy(mLMS2RGB.data, &LMS2RGB[0][0], bufsize);

	cv::Mat mLMS2lab1 = cv::Mat(3, 3, CV_64FC1);
	memcpy(mLMS2lab1.data, &LMS2lab1[0][0], bufsize);

	cv::Mat mLMS2lab2 = cv::Mat(3, 3, CV_64FC1);
	memcpy(mLMS2lab2.data, &LMS2lab2[0][0], bufsize);

	cv::Mat mLMS2lab = mLMS2lab2 * mLMS2lab1;
	cv::Mat mlab2LMS = mLMS2lab.inv();

	// Transform images from RGB to lab and
	// compute average and standard deviation of each color channels
	Color3d v;
	Color3d mt = Color3d(0.0, 0.0, 0.0);
	Color3d st = Color3d(0.0, 0.0, 0.0);
	for (int y = 0; y<target.rows; y++) {
		for (int x = 0; x<target.cols; x++) {
			v = target.at<Color3d>(y, x);
			v = mRGB2LMS * v;
			for (int c = 0; c<3; c++) v(c) = v(c) > eps ? log10(v(c)) : log10(eps);

			target.at<Color3d>(y, x) = mLMS2lab * v;
			mt = mt + target.at<Color3d>(y, x);
			st = st + target.at<Color3d>(y, x) * target.at<Color3d>(y, x);
		}
	}

	Color3d mr = Color3d(0.0, 0.0, 0.0);
	Color3d sr = Color3d(0.0, 0.0, 0.0);
	for (int y = 0; y<refer.rows; y++) {
		for (int x = 0; x<refer.cols; x++) {
			v = refer.at<Color3d>(y, x);
			v = mRGB2LMS * v;
			for (int c = 0; c<3; c++) v(c) = v(c) > eps ? log10(v(c)) : log10(eps);

			refer.at<Color3d>(y, x) = mLMS2lab * v;
			mr = mr + refer.at<Color3d>(y, x);
			sr = sr + refer.at<Color3d>(y, x) * refer.at<Color3d>(y, x);
		}
	}

	int Nt = target.rows * target.cols;
	int Nr = refer.rows * refer.cols;
	mt = mt.divide(Nt);
	mr = mr.divide(Nr);
	st = st.divide(Nt) - mt * mt;
	sr = sr.divide(Nr) - mr * mr;
	for (int i = 0; i<3; i++) {
		st(i) = sqrt(st(i));
		sr(i) = sqrt(sr(i));
	}

	// Transfer colors
	for (int y = 0; y<target.rows; y++) {
		for (int x = 0; x<target.cols; x++) {
			for (int c = 0; c<3; c++) {
				double val = target.at<double>(y, x * 3 + c);
				target.at<double>(y, x * 3 + c) = (val - mt(c)) / st(c) * sr(c) + mr(c);
			}
		}
	}


	// allocate host memory
	//h_C = new float[3]; //result

	float *h_C;
	int N = 3;
	h_C = (float*)malloc(sizeof(target.data));

	int rows = target.rows;
	int cols = target.cols;
	int size = sizeof(target.data);
	float* h_TARGET = (float *)malloc(sizeof(target.data));
	memcpy(h_TARGET, target.data, sizeof(target.data));
	matvec_L(&mlab2LMS, &mLMS2RGB, h_C, rows, cols, h_TARGET);
	//memcpy(&target.data, (unsigned char* )h_C, size);
	cv::Mat newTar = cv::Mat(rows, cols, CV_64FC3, (uchar*)h_C);

	// Transform back from lab to RGB
	/*for (int y = 0; y < target.rows; y++) {
	for (int x = 0; x < target.cols; x++) {
	v = target.at<Color3d>(y, x);

	//Not sure if all results are stored back into v so I'm leaving it as two seperate calls
	matvec_L(&v, &mlab2LMS, h_C);//Perform kernel launch and store result in h_C
	memcpy(&v, h_C, N * sizeof(float));// Transfer result back to v

	for (int c = 0; c < 3; c++)
	v(c) = v(c) > -5.0 ? pow(10.0, v(c)) : eps;

	matvec_L(&v, &mLMS2RGB, h_C);//Perform kernel launch and store result in h_C
	memcpy(&target.at<Color3d>(y, x), h_C, N * sizeof(float));// Transfer result back to v
	}
	}*/

	target.convertTo(newTar, CV_8UC3, 255.0);
	cv::cvtColor(newTar, newTar, CV_RGB2BGR);

	free(h_TARGET);

	//A2
	free(h_C);
	//delete[] h_C;

	cv::namedWindow("target");
	cv::imshow("target", newTar);
	cv::imwrite("output.jpg", newTar);
	cv::waitKey(0);
	cv::destroyAllWindows();
}


