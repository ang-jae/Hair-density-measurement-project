#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <malloc.h>
#include <iostream>
#include <cstdlib>
#include <cmath>

#include <opencv2/opencv.hpp>   
#include <opencv2/core/core.hpp>   
#include <opencv2/highgui/highgui.hpp> 
#include <opencv2/core.hpp>

using namespace cv;
using namespace std;

#define PI 3.14159265359

typedef struct {
	int r, g, b;
}int_rgb;

typedef struct {
	int h, s, v;
}int_hsv;


int** IntAlloc2(int height, int width)
{
	int** tmp;
	tmp = (int**)calloc(height, sizeof(int*));
	for (int i = 0; i < height; i++)
		tmp[i] = (int*)calloc(width, sizeof(int));
	return(tmp);
}

void IntFree2(int** image, int height, int width)
{
	for (int i = 0; i < height; i++)
		free(image[i]);

	free(image);
}

float** FloatAlloc2(int height, int width)
{
	float** tmp;
	tmp = (float**)calloc(height, sizeof(float*));
	for (int i = 0; i < height; i++)
		tmp[i] = (float*)calloc(width, sizeof(float));
	return(tmp);
}

void FloatFree2(float** image, int height, int width)
{
	for (int i = 0; i < height; i++)
		free(image[i]);

	free(image);
}

int_rgb** IntColorAlloc2(int height, int width)
{
	int_rgb** tmp;
	tmp = (int_rgb**)calloc(height, sizeof(int_rgb*));
	for (int i = 0; i < height; i++)
		tmp[i] = (int_rgb*)calloc(width, sizeof(int_rgb));
	return(tmp);
}

void IntColorFree2(int_rgb** image, int height, int width)
{
	for (int i = 0; i < height; i++)
		free(image[i]);

	free(image);
}

int** ReadImage(char* name, int* height, int* width)
{
	Mat img = imread(name, IMREAD_GRAYSCALE);
	int** image = (int**)IntAlloc2(img.rows, img.cols);

	*width = img.cols;
	*height = img.rows;

	for (int i = 0; i < img.rows; i++)
		for (int j = 0; j < img.cols; j++)
			image[i][j] = img.at<unsigned char>(i, j);

	return(image);
}

void WriteImage(char* name, int** image, int height, int width)
{
	Mat img(height, width, CV_8UC1);
	for (int i = 0; i < height; i++)
		for (int j = 0; j < width; j++)
			img.at<unsigned char>(i, j) = (unsigned char)image[i][j];

	imwrite(name, img);
}


void ImageShow(char* winname, int** image, int height, int width)
{
	Mat img(height, width, CV_8UC1);
	for (int i = 0; i < height; i++)
		for (int j = 0; j < width; j++)
			img.at<unsigned char>(i, j) = (unsigned char)image[i][j];
	imshow(winname, img);
	waitKey(0);
}



int_rgb** ReadColorImage(char* name, int* height, int* width)
{
	Mat img = imread(name, IMREAD_COLOR);
	int_rgb** image = (int_rgb**)IntColorAlloc2(img.rows, img.cols);

	*width = img.cols;
	*height = img.rows;

	for (int i = 0; i < img.rows; i++)
		for (int j = 0; j < img.cols; j++) {
			image[i][j].b = img.at<Vec3b>(i, j)[0];
			image[i][j].g = img.at<Vec3b>(i, j)[1];
			image[i][j].r = img.at<Vec3b>(i, j)[2];
		}

	return(image);
}

void WriteColorImage(char* name, int_rgb** image, int height, int width)
{
	Mat img(height, width, CV_8UC3);
	for (int i = 0; i < height; i++)
		for (int j = 0; j < width; j++) {
			img.at<Vec3b>(i, j)[0] = (unsigned char)image[i][j].b;
			img.at<Vec3b>(i, j)[1] = (unsigned char)image[i][j].g;
			img.at<Vec3b>(i, j)[2] = (unsigned char)image[i][j].r;
		}

	imwrite(name, img);
}

void ColorImageShow(char* winname, int_rgb** image, int height, int width)
{
	Mat img(height, width, CV_8UC3);
	for (int i = 0; i < height; i++)
		for (int j = 0; j < width; j++) {
			img.at<Vec3b>(i, j)[0] = (unsigned char)image[i][j].b;
			img.at<Vec3b>(i, j)[1] = (unsigned char)image[i][j].g;
			img.at<Vec3b>(i, j)[2] = (unsigned char)image[i][j].r;
		}
	imshow(winname, img);

}

template <typename _TP>
void ConnectedComponentLabeling(_TP** seg, int height, int width, int** label, int* no_label)
{

	//Mat bw = threshval < 128 ? (img < threshval) : (img > threshval);
	Mat bw(height, width, CV_8U);

	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++)
			bw.at<unsigned char>(i, j) = (unsigned char)seg[i][j];
	}
	Mat labelImage(bw.size(), CV_32S);
	*no_label = connectedComponents(bw, labelImage, 8); // 0까지 포함된 갯수임

	(*no_label)--;

	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++)
			label[i][j] = labelImage.at<int>(i, j);
	}
}

#define imax(x, y) ((x)>(y) ? x : y)
#define imin(x, y) ((x)<(y) ? x : y)

int BilinearInterpolation(int**image, int width, int height, double xx, double yy)
{
	int A_x = (int)xx; int A_y = (int)yy;
	int B_x = A_x + 1; int B_y = A_y;
	int C_x = A_x; int C_y = A_y + 1;
	int D_x = A_x + 1; int D_y = A_y + 1;


#if 0
	A_x = imin(imax(A_x, 0), width - 1);	B_x = imin(imax(B_x, 0), width - 1);	C_x = imin(imax(C_x, 0), width - 1);	D_x = imin(imax(D_x, 0), width - 1);
	A_y = imin(imax(A_y, 0), height - 1);	B_y = imin(imax(B_y, 0), height - 1);	C_y = imin(imax(C_y, 0), height - 1);	D_y = imin(imax(D_y, 0), height - 1);

#else
	if (A_x<0 || A_x > width - 1 || B_x<0 || B_x > width - 1 || C_x<0 || C_x > width - 1 || D_x<0 || D_x > width - 1
		|| A_y<0 || A_y > height - 1 || B_y<0 || B_y > height - 1 || C_y<0 || C_y > height - 1 || D_y<0 || D_y > height - 1)
		return (0);

#endif

	double dx = xx - A_x;
	double dy = yy - A_y;

	int X = image[A_y][A_x] * (1 - dx) * (1 - dy) + image[B_y][B_x] * dx * (1 - dy) + image[C_y][C_x] * (1 - dx) * dy + image[D_y][D_x] * dx * dy;

	return X;
}

#define SQ(x) ((x)*(x))

void ImageShowFloat(char* winname, float** image, int height, int width)
{
	Mat img(height, width, CV_8UC1);
	for (int i = 0; i < height; i++)
		for (int j = 0; j < width; j++)
			img.at<unsigned char>(i, j) = (unsigned char)image[i][j];
	imshow(winname, img);
	waitKey(0);
}

void ResizeImage(int**image, int**image_out, int height, int width, int height_out, int width_out)
{
	for (int i = 0; i < height_out; i++)
	{
		for (int j = 0; j < width_out; j++)
		{
			int x = j * ((float)width / (float)width_out);
			int y = i * ((float)height / (float)height_out);
			if (j < 0 || j >= width_out || i < 0 || i >= height_out || x < 0 || x >= width || y < 0 || y >= height) continue;
			else image_out[i][j] = image[y][x];//BilinearInterpolation(image, width, height, x, y);
		}
	}
}

void DownSize2(int**image, int height, int width, int**image_out)
{
	int height_out = height / 2, width_out = width / 2;
	for (int i = 0; i < height; i += 2)
	{
		for (int j = 0; j < width; j += 2)
		{
			image_out[(int)i / 2][(int)j / 2] = (image[i][j] + image[i][j + 1] + image[i + 1][j] + image[i + 1][j + 1]) / 4;
		}
	}
}

void DownSizeN(int**image, int height, int width, int n, int**image_out)
{
	int height_out = height / n, width_out = width / n;
	for (int i = 0; i < height; i += n)
	{
		for (int j = 0; j < width; j += n)
		{
			for (int k = 0; k < n; k++)
			{
				for (int l = 0; l < n; l++)
				{
					image_out[(int)i / n][(int)j / n] += image[i + k][j + l];
				}
			}
			image_out[(int)i / n][(int)j / n] /= n * n;


		}
	}
}

void GeometricTransform(int**block, int**block_changed, int height_b, int width_b, int num)
//num 0:가만히 1:x축대칭 2:y축대칭 3:원점대칭 4:대각선대칭(/) 5:대각선대칭(반대) 6:90도 회전 7:180도 회전
{
	//	scanf_s("%d", &num);
	switch (num) {

	case 0:
		for (int i = 0; i < height_b; i++) {
			for (int j = 0; j < width_b; j++) {
				block_changed[i][j] = block[i][j];
			}
		}
		break;
	case 1:
		for (int i = 0; i < height_b; i++) {
			for (int j = 0; j < width_b; j++) {
				block_changed[i][j] = block[height_b - 1 - i][j];
			}
		}
		break;
	case 2:
		for (int i = 0; i < height_b; i++) {
			for (int j = 0; j < width_b; j++) {
				block_changed[i][j] = block[i][width_b - 1 - j];
			}
		}
		break;
	case 3:
		for (int i = 0; i < height_b; i++) {
			for (int j = 0; j < width_b; j++) {
				block_changed[i][j] = block[j][i];
			}
		}
		break;
	case 4:
		for (int i = 0; i < height_b; i++) {
			for (int j = 0; j < width_b; j++) {
				block_changed[i][j] = block[width_b - 1 - j][height_b - 1 - i];
			}
		}
		break;
	case 5:
		for (int i = 0; i < height_b; i++) {
			for (int j = 0; j < width_b; j++) {
				block_changed[i][j] = block[j][height_b - 1 - i];
			}
		}
		break;
	case 6:
		for (int i = 0; i < height_b; i++) {
			for (int j = 0; j < width_b; j++) {
				block_changed[i][j] = block[height_b - 1 - i][width_b - 1 - j];
			}
		}
		break;
	case 7:
		for (int i = 0; i < height_b; i++) {
			for (int j = 0; j < width_b; j++) {
				block_changed[i][j] = block[height_b - 1 - j][i];
			}
		}
		break;
	default:
		printf("\nWrong number");
		break;
	}
}

void LightenImage(int**image, int height, int width, int num)
{
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			if (image[i][j] > 255) image[i][j] = 255;
			else image[i][j] += num;
		}
	}
}

int GetBlockAvg(int** image, int height, int width, int y, int x, int N)
{
	int avg = 0;
	for (int i = 0; i < N; i++)
	{
		for (int j = 0; j < N; j++)
		{
			if (y + i >= height || x + j >= width) continue;
			else avg += image[y + i][x + j];
		}
	}
	avg /= (N * N);
	return avg;
}

int RemoveMean(int** block, int N, int** block_mean)
{
	int mean = GetBlockAvg(block, N, N, 0, 0, N);
	for (int i = 0; i < N; i++)
	{
		for (int j = 0; j < N; j++)
		{
			block_mean[i][j] = block[i][j] - mean;
		}
	}
	return mean;
}

void ReadBlock(int**image, int height_image, int width_image, int y, int x, int dy, int dx, int**block)
{
	for (int i = 0; i < dy; i++)
	{
		for (int j = 0; j < dx; j++)
		{
			if (y + i >= height_image || x + j >= width_image || y + i < 0 || x + j < 0) continue;
			else block[i][j] = image[y + i][x + j];
		}
	}
}

void WriteBlock(int**image, int height, int width, int y, int x, int dy, int dx, int**block)
{
	for (int i = 0; i < dy; i++)
	{
		for (int j = 0; j < dx; j++)
		{
			if (y + i >= height || x + j >= width) continue;
			else image[y + i][x + j] = block[i][j];
		}
	}
}

void Scaling(int**image_input, float alpha, int dy, int dx, int**image_scaled)
{
	for (int i = 0; i < dy; i++)
	{
		for (int j = 0; j < dx; j++)
		{
			image_scaled[i][j] = alpha * (float)image_input[i][j] + 0.5;
		}
	}
}

void CopyImage(int**image, int height, int width, int**image_copied)
{
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			image_copied[i][j] = image[i][j];
		}
	}
}


/*
머리카락 프로젝트
*/

// COLOR_RGB2HSV


/*! \brief Convert RGB to HSV color space

Converts a given set of RGB values `r', `g', `b' into HSV
coordinates. The input RGB values are in the range [0, 1], and the
output HSV values are in the ranges h = [0, 360], and s, v = [0,
1], respectively.

\param fR Red component, used as input, range: [0, 1]
\param fG Green component, used as input, range: [0, 1]
\param fB Blue component, used as input, range: [0, 1]
\param fH Hue component, used as output, range: [0, 360]
\param fS Hue component, used as output, range: [0, 1]
\param fV Hue component, used as output, range: [0, 1]

*/
int_hsv RGBtoHSV(float fR, float fG, float fB) {
	int_hsv output;
	float fH, fS, fV;
	float fCMax = max(max(fR, fG), fB);
	float fCMin = min(min(fR, fG), fB);
	float fDelta = fCMax - fCMin;

	if (fDelta > 0) {
		if (fCMax == fR) {
			fH = 30 * (fmod(((fG - fB) / fDelta), 6));
		}
		else if (fCMax == fG) {
			fH = 30 * (((fB - fR) / fDelta) + 2);
		}
		else if (fCMax == fB) {
			fH = 30 * (((fR - fG) / fDelta) + 4);
		}

		if (fCMax > 0) {
			fS = fDelta / fCMax;
		}
		else {
			fS = 0;
		}

		fV = fCMax;
	}		
	else {
		fH = 0;
		fS = 0;
		fV = fCMax;
	}

	if (fH < 0) {
		fH = 180 + fH;
	}
	output.h = imax(imin(fH, 180), 0);
	output.s = imax(imin(fS * 255, 255), 0);
	output.v = imax(imin(fV * 255, 255), 0);
	return output;
}


/*! \brief Convert HSV to RGB color space

Converts a given set of HSV values `h', `s', `v' into RGB
coordinates. The output RGB values are in the range [0, 1], and
the input HSV values are in the ranges h = [0, 360], and s, v =
[0, 1], respectively.

\param fR Red component, used as output, range: [0, 1]
\param fG Green component, used as output, range: [0, 1]
\param fB Blue component, used as output, range: [0, 1]
\param fH Hue component, used as input, range: [0, 360]
\param fS Hue component, used as input, range: [0, 1]
\param fV Hue component, used as input, range: [0, 1]

*/

int_rgb HSVtoRGB(float fH, float fS, float fV) {
	int_rgb output;
	float fR, fG, fB;
	float fC = fV * fS; // Chroma
	float fHPrime = fmod(fH / 60.0, 6);
	float fX = fC * (1 - fabs(fmod(fHPrime, 2) - 1));
	float fM = fV - fC;

	if (0 <= fHPrime && fHPrime < 1) {
		fR = fC;
		fG = fX;
		fB = 0;
	}
	else if (1 <= fHPrime && fHPrime < 2) {
		fR = fX;
		fG = fC;
		fB = 0;
	}
	else if (2 <= fHPrime && fHPrime < 3) {
		fR = 0;
		fG = fC;
		fB = fX;
	}
	else if (3 <= fHPrime && fHPrime < 4) {
		fR = 0;
		fG = fX;
		fB = fC;
	}
	else if (4 <= fHPrime && fHPrime < 5) {
		fR = fX;
		fG = 0;
		fB = fC;
	}
	else if (5 <= fHPrime && fHPrime < 6) {
		fR = fC;
		fG = 0;
		fB = fX;
	}
	else {
		fR = 0;
		fG = 0;
		fB = 0;
	}

	fR += fM;
	fG += fM;
	fB += fM;

	output.r = fR;
	output.g = fG;
	output.b = fB;
	return output;
}

int_hsv** IntHSVAlloc2(int height, int width)
{
	int_hsv** tmp;
	tmp = (int_hsv**)calloc(height, sizeof(int_hsv*));
	for (int i = 0; i < height; i++)
		tmp[i] = (int_hsv*)calloc(width, sizeof(int_hsv));
	return(tmp);
}

void RGBimg_to_HSVimg(int_rgb**image, int_hsv**image_hsv, int height, int width)
{
	float H, S, V;
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			image_hsv[i][j] = RGBtoHSV((float)image[i][j].r, (float)image[i][j].g, (float)image[i][j].b);
		}
	}
}

void HSVImageShow(char* winname, int_hsv** image_hsv, int height, int width)
{
	Mat img(height, width, CV_8UC3);
	for (int i = 0; i < height; i++)
		for (int j = 0; j < width; j++) {
			img.at<Vec3b>(i, j)[0] = (unsigned char)image_hsv[i][j].h;
			img.at<Vec3b>(i, j)[1] = (unsigned char)image_hsv[i][j].s;
			img.at<Vec3b>(i, j)[2] = (unsigned char)image_hsv[i][j].v;
		}
	imshow(winname, img);

}

void ShowHue(char* winname, int_hsv** image_hsv, int height, int width)
{
	Mat img(height, width, CV_8UC3);
	for (int i = 0; i < height; i++)
		for (int j = 0; j < width; j++) {
			img.at<Vec3b>(i, j)[0] = (unsigned char)image_hsv[i][j].h;
			img.at<Vec3b>(i, j)[1] = 0;
			img.at<Vec3b>(i, j)[2] = 0;
		}
	imshow(winname, img);
}

void ShowSaturation(char* winname, int_hsv** image_hsv, int height, int width)
{
	Mat img(height, width, CV_8UC3);
	for (int i = 0; i < height; i++)
		for (int j = 0; j < width; j++) {
			img.at<Vec3b>(i, j)[0] = 0;
			img.at<Vec3b>(i, j)[1] = (unsigned char)image_hsv[i][j].s;
			img.at<Vec3b>(i, j)[2] = 0;
		}
	imshow(winname, img);
}

void ShowValue(char* winname, int_hsv** image_hsv, int height, int width)
{
	Mat img(height, width, CV_8UC3);
	for (int i = 0; i < height; i++)
		for (int j = 0; j < width; j++) {
			img.at<Vec3b>(i, j)[0] = 0;
			img.at<Vec3b>(i, j)[1] = 0;
			img.at<Vec3b>(i, j)[2] = (unsigned char)image_hsv[i][j].v;
		}
	imshow(winname, img);
}

int Lowerbound_H = 0;
int Upperbound_H = 21;
float Lowerbound_S = 0.2;
float Upperbound_S = 0.6;
int Lowerbound_V = 80;
int Upperbound_V = 255;

Mat ExtractSkin(char* winname, Mat &image, Mat &image_h, Mat &image_s, Mat &image_v ,int height, int width)
{
	Mat img(height, width, CV_8UC3);
	if (image_h.empty() || image_s.empty() || image_v.empty())
	{
		printf("\nWrong input!");
		return img;
	}
	for (int i = 0 ; i < height; i++)
		for (int j = 0 ; j < width ; j++) {
			if (image_h.at<Vec3b>(i, j)[0] >= Lowerbound_H && image_h.at<Vec3b>(i, j)[0] <= Upperbound_H
				&& image_s.at<Vec3b>(i, j)[1] >= Lowerbound_S && image_s.at<Vec3b>(i, j)[1] <= Upperbound_S
				&& image_v.at<Vec3b>(i, j)[2] >= Lowerbound_V && image_v.at<Vec3b>(i, j)[2] <= Upperbound_V
				)
			{
				img.at<Vec3b>(i, j)[0] = image.at<Vec3b>(i, j)[0];
				img.at<Vec3b>(i, j)[1] = image.at<Vec3b>(i, j)[1];
				img.at<Vec3b>(i, j)[2] = image.at<Vec3b>(i, j)[2];
			}
			else
			{
				img.at<Vec3b>(i, j)[0] = 0;
				img.at<Vec3b>(i, j)[1] = 0;
				img.at<Vec3b>(i, j)[2] = 0;
			}
			
		}
	return img;
}

/*
int main_(int argc, char** argv) {
	float fR = 0, fG = 0, fB = 0, fH = 0, fS = 0, fV = 0;

	fH = 146.0;
	fS = 0.19;
	fV = 0.66;

	HSVtoRGB(fR, fG, fB, fH, fS, fV);

	fR = 136.0;
	fG = 168.0;
	fB = 150.0;

	RGBtoHSV(fR, fG, fB, fH, fS, fV);

	cout << "[RGB] "
		<< "Float:   (" << fR << ", " << fG << ", " << fB << ")" << endl
		<< "      Integer: (" << (255 * fR) << ", " << (255 * fG) << ", " << (255 * fB) << ")" << endl;
	cout << "[HSV] (" << fH << ", " << fS << ", " << fV << ")" << endl;

	return EXIT_SUCCESS;
}
*/


//https://s-engineer.tistory.com/139

int lowerHue = 40, upperHue = 80; //green
Mat src, src_hsv, mask, dst;

void OnHueChanged(int pos, void* userdata)
{
	Scalar lowerb(lowerHue, 100, 0);
	Scalar upperb(upperHue, 255, 255);

	inRange(src_hsv, lowerb, upperb, mask);

	dst.setTo(0);
	src.copyTo(dst, mask);
	imshow("mask", mask);
	imshow("dst", dst);
}

void _main(int argc, char** argv)
{
	src = imread("hair7.jpg", IMREAD_COLOR);
	if (src.empty()) {
		cerr << "Image load failed." << endl;
	}

	imshow("src", src);

	cvtColor(src, src_hsv, COLOR_BGR2HSV);

	namedWindow("mask");
	createTrackbar("Lower Hue", "mask", &lowerHue, 179, OnHueChanged);
	createTrackbar("Upper Hue", "mask", &upperHue, 179, OnHueChanged);
	OnHueChanged(NULL, NULL);

	waitKey(0);
}
/*
void main_(int argc, char** argv)
{
	int height, width;
	int_rgb** image = ReadColorImage((char*)"hair3.jpg", &height, &width);
	int_hsv** image_hsv = IntHSVAlloc2(height, width);
	
	RGBimg_to_HSVimg(image, image_hsv, height, width);

	ColorImageShow("Original", image, height, width);
	HSVImageShow("HSV", image_hsv, height, width);
	ShowHue("Hue", image_hsv, height, width);
	ShowSaturation("Saturation", image_hsv, height, width);
	ShowValue("Value", image_hsv, height, width);
	ShowSkin("Skin", image, image_hsv, height, width);
	waitKey(0);
}
*/
void main_0713()
{
	Mat img_hsv, img_rgb;
	img_rgb = imread("hair7.jpg", 1);
	cvtColor(img_rgb, img_hsv, COLOR_BGR2HSV);
	Mat hsv_images[3];
	Mat img_skin;
	split(img_hsv, hsv_images);

	namedWindow("win1", WINDOW_AUTOSIZE);
	imshow("h", hsv_images[0]);
	imshow("s", hsv_images[1]);
	imshow("v", hsv_images[2]);
//	img_skin = ExtractSkin("Skin", img_rgb, hsv_images[0], hsv_images[1], hsv_images[2], img_rgb.rows, img_rgb.cols);

	/*
	src = imread("hair7.jpg", IMREAD_COLOR);
	if (src.empty()) {
		cerr << "Image load failed." << endl;
	}

	imshow("src", src);

	cvtColor(src, src_hsv, COLOR_BGR2HSV);

	namedWindow("mask");
	createTrackbar("Lower Hue", "mask", &lowerHue, 179, OnHueChanged);
	createTrackbar("Upper Hue", "mask", &upperHue, 179, OnHueChanged);
	OnHueChanged(NULL, NULL);
	*/

	waitKey(0);
}

float DensityMeasurement(Mat &image, int height, int width)
{
	float density;
	int count = 0; // hair 점 개수

	for(int i = 0; i < height ; i++)
		for (int j = 0; j < width ; j++)
		{
			if (image.at<Vec3b>(i, j) == Vec3b(0, 0, 0)) count++; // 여기서 오류발생
		}
	//머리카락 밀도 = hair 점 개수 / row*col
	density = count / (height*width);

	return density;
}

void main() // YCbCr로 변경
{
	Mat img_rgb, img_YCrCb, img_skin;
	//황색은 Cb : 77 ~ 127, Cr : 133 ~ 173. 조금씩 조정
	int upperb_Cr = 173;
	int lowerb_Cr = 133;
	int upperb_Cb = 130;
	int lowerb_Cb = 70;


	img_rgb = imread("hair7.jpg", 1);
	cvtColor(img_rgb, img_YCrCb, COLOR_BGR2YCrCb);
	inRange(img_YCrCb, Scalar(0, lowerb_Cr, lowerb_Cb), Scalar(255, upperb_Cr, upperb_Cb), img_YCrCb);

	img_skin = (img_YCrCb.size(), CV_8UC3, Scalar(0));
	add(img_rgb, Scalar(0), img_skin, img_YCrCb);

	imshow("Original", img_rgb);
	imshow("Changed Image", img_YCrCb);
	imshow("Skin", img_skin);

	float density = DensityMeasurement(img_YCrCb, img_YCrCb.rows, img_YCrCb.cols);
	printf("\nDensity is : %f", density);

	waitKey(0);
}

