#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <malloc.h>

#include <opencv2/opencv.hpp>   
#include <opencv2/core/core.hpp>   
#include <opencv2/highgui/highgui.hpp>  

using namespace cv;

#define PI 3.14159265359

typedef struct {
	int r, g, b;
}int_rgb;


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

int FindErr(int**image, int**block, int x, int y, int width, int height, int width_t, int height_t)
{
	int Err = 0;
	for (int i = 0; i < height_t; i++)
	{
		for (int j = 0; j < width_t; j++)
		{
			if (x + j >= width || y + i >= height)continue;
			else Err += abs(image[y + i][x + j] - block[i][j]);
		}
	}
	return Err;
}

int FindErrRMS(int**image, int**block, int x, int y, int width, int height, int width_t, int height_t)
{
	int Err = 0;
	for (int i = 0; i < height_t; i++)
	{
		for (int j = 0; j < width_t; j++)
		{
			if (x + j >= width || y + i >= height)continue;
			else Err += (image[y + i][x + j] - block[i][j])*(image[y + i][x + j] - block[i][j]);
		}
	}
	Err /= height_t * width_t;
	return (sqrt(Err));
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
void CoverImageWithTemplate(int**image, int**block, int height, int width, int height_b, int width_b, POS2D pos)
{
	for (int i = 0; i < height_b; i++)
	{
		for (int j = 0; j < width_b; j++)
		{
			if ((int)pos.y + i >= height || (int)pos.x + j >= width || (int)pos.y + i < 0 || (int)pos.x + j < 0) continue;
			else image[(int)pos.y + i][(int)pos.x + j] = block[i][j];
		}
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

char RGB_to_HSV()
{

}

void main()
{


}