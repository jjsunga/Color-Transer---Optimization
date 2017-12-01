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
#include <chrono>
#include <cstdlib>
using namespace std::chrono;

#include <opencv2\opencv.hpp>

//CUDA
#include "cuda_runtime.h"
#include <cublas_v2.h>
#include <device_launch_parameters.h>
#include "Color3d.h"

//A2 START
#define BLOCK_SIZE 512

void reportTime(const char* msg, steady_clock::duration span) {
	auto ms = duration_cast<milliseconds>(span);
	std::cout << msg << " - took - " <<
		ms.count() << " millisecs" << std::endl;
}

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

/*
__device__ void matMulMat(float* A, float* B, float* c)
{
for (int x = 0; x < 3; ++c)
{
for (int y = 0; y < 3; ++y)
{
float sum = 0.0f;
for (int z = 0; z < 3; ++z)
{
sum = a[i * 3 + z] * b[i * 3 + z];
}

c[i * 3 + y] = sum;
}
}
}

__device__ void matSubMat(float* A, float* B, float* c)
{
for (int x = 0; x < 3; ++c)
{
for (int y = 0; y < 3; ++y)
{
float sum = 0.0f;
for (int z = 0; z < 3; ++z)
{
sum = a[i * 3 + z] - b[i * 3 + z];
}

c[i * 3 + y] = sum;
}
}
}

__device__ void matDivSca(float* A, float* B, float* c)
{
float sum = 0.0f;
for (int i = 0; i < 3; ++i)
{
for (int j = 0; j < 3; ++j)
{
c[i* 3 + j] = a[i * 3 + j] / b;
}
}
}

__device__ void matSubSca(float* A, float* B, float* c)
{
float sum = 0.0f;
for (int i = 0; i < 3; ++i)
{
for (int j = 0; j < 3; ++j)
{
c[i* 3 + j] = a[i * 3 + j] - b;
}
}
}

__device__ void vecSubSca(float* A, float* B, float* c)
{
for (int i = 0; i < 3; ++i)
{
c[i] = a[i] - b;
}
}

__device__ void vecPluVec(float* A, float* B, float* c)
{
for (int i = 0; i < 3; ++i)
{
c[i] = a[i] + b[i];
}
}

__device__ void vecDivVec(float* A, float* B, float* c)
{
for (int i = 0; i < 3; ++i)
{
c[i] = a[i] / b[i];
}
}

__device__ void vecDivVec(float* A, float* B, float* c)
{
for (int i = 0; i < 3; ++i)
{
c[i] = a[i] * b[i];
}
}
*/

__global__ void matvec_kernel(float* d_A, float* d_RGB2, float* d_LMS2, float* d_C,
	const int n, int targetrows, int targetcols, float* d_Tar)//,
															  //                              float* mt, float* mr, float* st, float* sr, float* temp1, float* temp2)
{
	/*
	int Nt = targetrows * targetcols;
	int Nr = referrows * refercols;

	matDivSca(mt, &Nt, temp1);
	memcpy(&mt, &temp1, 3 * 3 * sizeof(float));
	matDivSca(mr, &Nr, temp1);
	memcpy(&mr, &temp1, 3 * 3 * sizeof(float));

	matDivSca(st, &Nt, temp1);
	matMulMat(mt, mt, temp2);
	matSubMat(temp1, temp2, st);

	matDivSca(sr, &Nr, temp1);
	matMulMat(mr, mr, temp2);
	matSubMat(temp1, temp2, sr);

	for (int i = 0; i < 3; i++)
	{
	st[i] = sqrt(st[i]);
	sr[i] = sqrt(sr[i]);
	}

	// Transfer colors
	//grid-stride loop
	for(int tid = threadIdx.x + blockIdx.x * blockDim.x;
	tid < targetrows;
	tid += blockDim.x * gridDim.x)
	{
	for (int x = 0; x < targetcols; x++)
	{
	for (int y = 0; y < 3; c++)
	{
	double val = d_Tar[tid + 3 (x + 3 * y)];

	vecSubSca(mt[y], val, temp1);
	vecDivVec(temp1, st[y], temp2);
	vecMulVec(temp2, sr[y], temp1);
	vecPluVec(temp1, mr[y], d_Tar[tid + 3 (x + 3 * y)]);
	}
	}
	}
	*/
	//int tid = threadIdx.x + blockIdx.x * blockDim.x;
	const double eps = 1.0e-4;
	//grid-stride loop
	for (int tid = threadIdx.x + blockIdx.x * blockDim.x;
		tid < targetrows;
		tid += blockDim.x * gridDim.x)
		//for (int y = 0; y < targetrows; ++y)
		//if(tid < targetrows)
	{
		for (int x = 0; x < targetcols; ++x) {
			memcpy(&d_A, &d_Tar[tid * 3 + x], 3 * sizeof(float));

			matvec(d_A, d_RGB2, d_C);
			memcpy(&d_A, d_C, 3 * sizeof(float));

			for (int c = 0; c < 3; c++)
				d_A[c] = d_A[c] > -5.0 ? __powf(10.0f, d_A[c]) : eps;

			matvec(d_A, d_LMS2, d_C);
			memcpy(&d_Tar[tid * 3 + x], d_C, 3 * sizeof(float));
		}
	}
}

inline void vecTransfer(float* h, Color3d* v)
{
	for (int j = 0; j < 3; ++j)
		h[j] = v->v[j];
}

//KERNEL Helper function does setup and launch
void matvec_L(cv::Mat* mRGB2LMS, cv::Mat* mLMS2lab, float* h_C, int tarrow, int tarcol, float* h_Tar)//,
																									 //              float* mt, float* mr, float* st, float* sr)
{
	float *h_A, *h_RGB2, *h_LMS2, *d_Tar;//, *h_mt, *h_mr,*h_st, *h_sr;
	float *d_A, *d_RGB2, *d_LMS2, *d_C;//, *temp1, *temp2, *d_mt, *d_mr,*d_st, *d_sr;

	int N = 3;

	h_A = (float*)malloc(sizeof(float) * N);
	h_RGB2 = new float[mRGB2LMS->total()];
	h_LMS2 = new float[mLMS2lab->total()];
	//    h_mt = new float[mt->total()];
	//    h_mr = new float[mr->total()];
	//    h_st = new float[st->total()];
	//    h_sr = new float[sr->total()];

	//h_C = (float*)malloc(sizeof(float) * N);

	cudaMalloc((void**)&d_A, sizeof(float) * N);
	cudaMalloc((void**)&d_RGB2, sizeof(float) * N * N);
	cudaMalloc((void**)&d_LMS2, sizeof(float) * N * N);
	cudaMalloc((void**)&d_C, sizeof(h_C));
	cudaMalloc((void**)&d_Tar, sizeof(h_Tar));
	//    cudaMalloc((void**)&temp1, sizeof(float) * N * N);
	//    cudaMalloc((void**)&temp2, sizeof(float) * N * N);
	//    cudaMalloc((void**)&d_mt, sizeof(float) * N * N);
	//    cudaMalloc((void**)&d_mr, sizeof(float) * N * N);
	//    cudaMalloc((void**)&d_st, sizeof(float) * N * N);
	//    cudaMalloc((void**)&d_sr, sizeof(float) * N * N);
	Color3d vec;

	//copy vec and matrix to host pointers
	vecTransfer(h_A, &vec);
	memcpy(h_RGB2, mRGB2LMS->data, mRGB2LMS->total());
	memcpy(h_LMS2, mLMS2lab->data, mLMS2lab->total());

	//    memcpy(h_mt, mt->data, mt->total());
	//    memcpy(h_mr, mr->data, mr->total());
	//    memcpy(h_st, st->data, st->total());
	//    memcpy(h_sr, sr->data, sr->total());

	cudaMemcpy(d_A, h_A, sizeof(float) * N, cudaMemcpyHostToDevice);
	cudaMemcpy(d_RGB2, h_RGB2, sizeof(float) * N * N, cudaMemcpyHostToDevice);
	cudaMemcpy(d_LMS2, h_LMS2, sizeof(float) * N * N, cudaMemcpyHostToDevice);
	cudaMemcpy(d_Tar, h_Tar, sizeof(h_Tar), cudaMemcpyHostToDevice);

	//    cudaMemcpy(d_mt, h_mt, sizeof(float) * N * N, cudaMemcpyHostToDevice);
	//    cudaMemcpy(d_mr, h_mr, sizeof(float) * N * N, cudaMemcpyHostToDevice);
	//    cudaMemcpy(d_st, h_st, sizeof(float) * N * N, cudaMemcpyHostToDevice);
	//    cudaMemcpy(d_sr, h_sr, sizeof(float) * N * N, cudaMemcpyHostToDevice);

	matvec_kernel << <N / BLOCK_SIZE + 1, BLOCK_SIZE >> >(d_A, d_RGB2, d_LMS2, d_C, N, tarrow, tarcol, d_Tar);//,
																											  //mt, mr, st, sr, temp1, temp2);
																											  //printf("error code: %s\n",cudaGetErrorString(cudaGetLastError()));

	cudaMemcpy(h_C, d_C, sizeof(h_C), cudaMemcpyDeviceToHost);

	free(h_A);
	free(h_RGB2);
	free(h_LMS2);
	//    free(h_mt);
	//    free(h_mr);
	//    free(h_st);
	//    free(h_sr);
	//free(h_C);

	cudaFree(d_A);
	cudaFree(d_RGB2);
	cudaFree(d_LMS2);
	cudaFree(d_C);
	cudaFree(d_Tar);
	//    cudaFree(temp1);
	//    cudaFree(temp2);
	//    cudaFree(d_mt);
	//    cudaFree(d_mr);
	//    cudaFree(d_st);
	//    cudaFree(d_sr);
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
		std::cout << "usage: > ColorTransfer.exe [target image] [reference image]" << std::endl;
		return -1;
	}



	// Load target image
	cv::Mat target = cv::imread(argv[1], CV_LOAD_IMAGE_COLOR);
	if (target.empty()) {
		std::cout << "Failed to load file \"" << argv[1] << "\"" << std::endl;
		return -1;
	}
	cv::cvtColor(target, target, CV_BGR2RGB);
	target.convertTo(target, CV_64FC3, 1.0 / 255.0);

	// Load reference image
	cv::Mat refer = cv::imread(argv[2], CV_LOAD_IMAGE_COLOR);
	if (refer.empty()) {
		std::cout << "Failed to load file \"" << argv[2] << "\"" << std::endl;
		return -1;
	}
	steady_clock::time_point ts, te;
	ts = steady_clock::now();

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
	te = steady_clock::now();
	reportTime("target conversion took", te - ts);
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