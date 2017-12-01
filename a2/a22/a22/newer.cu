
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

//A2 Added functions
#define BLOCK_SIZE 256

//KERNELS BEGIN - unoptimized version
//                            vec         mat         vec
__global__ void matvec_kernel(float* d_A, float* d_B, float* d_C, const int n)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;

	float sum = 0;
	if (tid < n)
	{
		for (int i = 0; i < n; ++i)
		{
			sum += d_A[i] * d_B[(i * n) + tid];
		}

		d_C[tid] = sum;
	}
}
//KERNELS END


void vecTransfer(float* h, Color3d* v)
{
	for (int j = 0; j < 3; ++j)
		h[j] = v->v[j]; //whats the point of making a class if you're just going to make everything public
}

//m.ptr<float>(0); // is a pointer to the beginning of the row in the matrix might be useful
void matTransfer(float* h, const cv::Mat& m)
{
	for (int i = 0; i < 3; ++i)
		for (int j = 0; i < 3; ++j)
			h[i * j + 3] = m.at<float>(i, j);
}

int math(int a, int b) {

	return a + b;
}

//KERNEL Helper function does setup and launch
void matvec_L(Color3d* vec, cv::Mat* mat, float* h_C)
{
	float *h_A, *h_B;
	float *d_A, *d_B, *d_C;

	int N = 3;
	h_A = (float*)malloc(sizeof(float) * N);
	h_B = new float[mat->total()];
	//h_A = new float(N);
	//float* mPtr = new float[mat->total()];

	cudaMalloc((void**)&d_A, sizeof(float) * N);
	cudaMalloc((void**)&d_B, sizeof(float) * N * N);
	cudaMalloc((void**)&d_C, sizeof(float) * N);

	//copy vec and matrix to host pointers
	vecTransfer(h_A, vec);
	memcpy(h_B, mat->data, mat->total());



	cudaMemcpy(d_A, h_A, sizeof(float) * N, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, h_B, sizeof(float) * N * N, cudaMemcpyHostToDevice);

	matvec_kernel << <N / BLOCK_SIZE + 1, BLOCK_SIZE >> >(d_A, d_B, d_C, N);
	//printf("error code: %s\n",cudaGetErrorString(cudaGetLastError()));

	cudaMemcpy(h_C, d_C, sizeof(float) * N, cudaMemcpyDeviceToHost);

	free(h_A);
	free(h_B);
	//delete[] h_A;
	//delete[] h_B;

	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);


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

	//Easy transfer back
	//--------A2-----------
	float *h_C;
	int N = 3;
	h_C = (float*)malloc(sizeof(float) * N);
	//--------A2-----------

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
	for (int y = 0; y < target.rows; y++) {
		for (int x = 0; x < target.cols; x++) {
			v = target.at<Color3d>(y, x);

			matvec_L(&v, &mRGB2LMS, h_C);//Perform kernel launch and store result in h_C
			memcpy(&v, h_C, N * sizeof(float));// Transfer result back to v

			for (int c = 0; c < 3; c++)
				v(c) = v(c) > eps ? log10(v(c)) : log10(eps);

			matvec_L(&v, &mLMS2lab, h_C);//Perform kernel launch and store result in h_C
			memcpy(&target.at<Color3d>(y, x), h_C, N * sizeof(float));// Transfer result back to v

			mt = mt + target.at<Color3d>(y, x);
			st = st + target.at<Color3d>(y, x) * target.at<Color3d>(y, x);//maybe do this one too
		}
	}

	Color3d mr = Color3d(0.0, 0.0, 0.0);
	Color3d sr = Color3d(0.0, 0.0, 0.0);
	for (int y = 0; y < refer.rows; y++) {
		for (int x = 0; x < refer.cols; x++) {
			v = refer.at<Color3d>(y, x);

			matvec_L(&v, &mRGB2LMS, h_C);//Perform kernel launch and store result in h_C
			memcpy(&v, h_C, N * sizeof(float));// Transfer result back to v

			for (int c = 0; c < 3; c++)
				v(c) = v(c) > eps ? log10(v(c)) : log10(eps);

			matvec_L(&v, &mLMS2lab, h_C);//Perform kernel launch and store result in h_C
			memcpy(&refer.at<Color3d>(y, x), h_C, N * sizeof(float));// Transfer result back to v

			mr = mr + refer.at<Color3d>(y, x);
			sr = sr + refer.at<Color3d>(y, x) * refer.at<Color3d>(y, x);//maybe do this one too
		}
	}

	int Nt = target.rows * target.cols;
	int Nr = refer.rows * refer.cols;
	mt = mt.divide(Nt);
	mr = mr.divide(Nr);
	st = st.divide(Nt) - mt * mt;
	sr = sr.divide(Nr) - mr * mr;
	for (int i = 0; i < 3; i++) {
		st(i) = sqrt(st(i));
		sr(i) = sqrt(sr(i));
	}

	// Transfer colors
	for (int y = 0; y < target.rows; y++) {
		for (int x = 0; x < target.cols; x++) {
			for (int c = 0; c < 3; c++) {
				double val = target.at<double>(y, x * 3 + c);
				target.at<double>(y, x * 3 + c) = (val - mt(c)) / st(c) * sr(c) + mr(c);
			}
		}
	}

	// allocate host memory
	h_C = new float[3]; //result

						// Transform back from lab to RGB
	for (int y = 0; y < target.rows; y++) {
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
	}

	target.convertTo(target, CV_8UC3, 255.0);
	cv::cvtColor(target, target, CV_RGB2BGR);

	//A2
	//free(h_C);
	delete[] h_C;

	cv::namedWindow("target");
	cv::imshow("target", target);
	cv::imwrite("output.jpg", target);
	cv::waitKey(0);
	cv::destroyAllWindows();
}
