#include <stdio.h>
#include "highgui.h"
#include <opencv2\opencv.hpp>
#include <iostream>
#define VERTICAL  0
#define HORIZONTAL 1
#pragma comment( linker, "/subsystem:\"windows\" /entry:\"mainCRTStartup\"" ) 
using namespace cv;
void getEnergyMap(Mat src,Mat gradientMap) {
	//将原图像先转换成灰度图
	Mat grayImage(src.rows,src.cols,CV_8U,Scalar(0));
	cvtColor(src,grayImage,COLOR_RGB2GRAY);
	//imshow("gray",grayImage);
	//waitKey(0);
	Mat gradientH(src.rows,src.cols,CV_32F,Scalar(0));   //Horizontal gradient 水平梯度矩阵
	Mat gradientV(src.rows,src.cols,CV_32F,Scalar(0));   //Vertical gradient   竖直梯度矩阵
	//以下是用filter2D函数进行卷积 效率不是很高,但是结果正确
	Mat kernelH = (Mat_<float>(3,3) << 0,0,0,0,1,-1,0,0,0 );   //定义水平方向上的卷积核函数
	Mat kernelV = (Mat_<float>(3,3) << 0,0,0,0,1,0,0,-1,0);	   //定义竖直方向上的卷积核函数
	filter2D(grayImage,gradientH,gradientH.depth(),kernelH);  
	filter2D(grayImage,gradientV,gradientV.depth(),kernelV);  //分别对水平和竖直方向进行卷积

/*
	for (int j = 0; j < src.cols; j++) {
		gradientV.at<float>(src.rows - 1, j) = 0;
		if (j != src.cols - 1) {
			for (int i = 0; i < src.rows; i++) {
				gradientH.at<float>(i, j) = grayImage.at<char>(i, j) - grayImage.at<char>(i, j + 1);
			}
		}
	}
	for (int i = 0; i < src.rows; i++) {
		gradientH.at<float>(i, src.cols - 1) = 0;
		if (i != src.rows - 1) {
			for (int j = 0; j < src.cols; j++) {
				gradientV.at<float>(i, j) = grayImage.at<char>(i, j) - grayImage.at<char>(i + 1, j);
			}
		}
	}
*/
	add(abs(gradientH),abs(gradientV),gradientMap);    //将水平和竖直的滤波结果相加 可以得到近似梯度大小 存于矩阵gradientMap
	//Mat showGradient;
	//gradientMap.convertTo(showGradient,CV_8U,1,0);  //由于gradientMap的深度是cv_32f ,要显示出来首先转换成cv_8u
	//imshow("gradientShow",showGradient);
	//waitKey(0);     //这四行可以显示能量梯度图。
}
void calculateEnergy(Mat EnergyMap,Mat Accum,Mat traceMap,int code) {
//注：traceMap上每个像素点位置只放0，1，2三个数值  用来标志该像素点从何处继承而来。
	EnergyMap.copyTo(Accum);						//使用copyTo而不是= ，因为要复制整个矩阵而不仅仅是矩阵头。
	if (code == 0) {									//竖直方向的seam  traceMap中左上为0 正上为1 右上为2
		for (int i = 1; i < EnergyMap.rows; i++) {		//从第二行开始计算
			//第一列
			if (Accum.at<float>(i - 1, 0) <= Accum.at<float>(i - 1, 1)) {
				Accum.at<float>(i, 0) = Accum.at<float>(i - 1, 0) + EnergyMap.at<float>(i, 0);
				traceMap.at<char>(i, 0) = 1;
			}
			else {										//正上方像素点能量大于右边一列的情况 取右上角的点
				Accum.at<float>(i, 0) = Accum.at<float>(i - 1, 1) + EnergyMap.at<float>(i, 0);
				traceMap.at<char>(i, 0) = 2;
			}
			//中间列
			for (int j = 1; j < EnergyMap.cols -1; j++) {
				float k[3];
				k[0] = Accum.at<float>(i - 1, j - 1);
				k[1] = Accum.at<float>(i - 1, j);
				k[2] = Accum.at<float>(i - 1, j + 1);
				int index = 0;
				if (k[0] > k[1]) index = 1;
				if (k[2] < k[index]) index = 2;
				Accum.at<float>(i, j) = Accum.at<float>(i - 1, j - 1 + index) + EnergyMap.at<float>(i, j);
				traceMap.at<char>(i, j) = index;
			}
			//最后一列（最右边一列）
			if (Accum.at<float>(i - 1, EnergyMap.cols - 1) <= Accum.at<float>(i - 1, EnergyMap.cols - 2)) {
				Accum.at<float>(i, EnergyMap.cols - 1) = Accum.at<float>(i - 1, EnergyMap.cols - 1) + EnergyMap.at<float>(i, EnergyMap.cols - 1);
				traceMap.at<char>(i, EnergyMap.cols - 1) = 1;
			}
			else {  //从左上方继承的情况
				Accum.at<float>(i, EnergyMap.cols - 1) = Accum.at<float>(i - 1, EnergyMap.cols - 2) + EnergyMap.at<float>(i, EnergyMap.cols - 1);
				traceMap.at<char>(i, EnergyMap.cols - 1) = 0;
			}
		}
	}
	else if (code == 1) {  //从左边向右计算累计能量  寻找水平方向的seam 左上为0，正左边为1，左下为2
		for (int i = 1; i < EnergyMap.cols; i++) {	//从第二列开始计算
			//第一行
			if (Accum.at<float>(0, i - 1) <= Accum.at<float>(1, i - 1)) {		//左边的比左下的小，吸取左边的
				Accum.at<float>(0, i) = Accum.at<float>(0, i - 1) + EnergyMap.at<float>(0, i);
				traceMap.at<char>(0, i) = 1;
			}
			else {	//左下方的像素点更小 吸取左下角=2
				Accum.at<float>(0, i) = Accum.at<float>(1, i - 1) + EnergyMap.at<float>(0, i);
				traceMap.at<char>(0, i) = 2;
			}
			//中间行
			for (int j = 1; j < EnergyMap.rows - 1; j++) {
				float k[3];
				k[0] = Accum.at<float>(j - 1, i - 1);
				k[1] = Accum.at<float>(j, i - 1);
				k[2] = Accum.at<float>(j + 1, i - 1);
				int index = 0;
				if (k[0] > k[1]) index = 1;
				if (k[2] < k[index]) index = 2;
				Accum.at<float>(j, i) = Accum.at<float>(j - 1 + index, i - 1) + EnergyMap.at<float>(j, i);
				traceMap.at<char>(j, i) = index;
			}
			//最下面一行
			int c = EnergyMap.rows - 1;
			if (Accum.at<float>(c - 1, i - 1) <= Accum.at<float>(c, i - 1)) {	//左上方像素点更小
				Accum.at<float>(c, i) = Accum.at<float>(c - 1, i - 1) + EnergyMap.at<float>(c, i);
				traceMap.at<char>(c, i) = 0;
			}
			else {	//左方更小
				Accum.at<float>(c, i) = Accum.at<float>(c, i - 1) + EnergyMap.at<float>(c, i);
				traceMap.at<char>(c, i) = 1;
			}
		}
	}
}
void FindSeam(const Mat Acum,const Mat traceMap,Mat Seam,int code) {
	if (code == 0) {  //竖向的seam
		int row = Acum.rows - 1;
		int index = 0; //保存的是列标
		//下面标识出最后一行中的最小值
		for (int i = 1; i < Acum.cols; i++) {
			if (Acum.at<float>(row, i) < Acum.at<float>(row, index)) index = i;  //寻找最小能量值中靠近左边的
		}
		//下面根据traceMap向上回溯，找出minTrace
		Seam.at<float>(row, 0) = index;
		int pos = index; //记录在traceMap中当前的位置。
		for (int i = row; i > 0; i--) {
			if (traceMap.at<char>(i, pos) == 0) {		//向左走
				index -= 1;
			}
			else if (traceMap.at<char>(i, pos) == 2) {		//向右走
				index += 1;
			}
			Seam.at<float>(i - 1, 0) = index;
			pos = index;
		}
	}
	else if (code == 1) {	//横向的seam
		int col = Acum.cols - 1;
		int index = 0; //记录行标
		for (int i = 1; i < Acum.rows; i++) {
			if (Acum.at<float>(i, col) < Acum.at<float>(index, col)) index = i;
		}
		//由traceMap的最后一列向前回溯，找出横向的seam
		Seam.at<float>(0, col) = index;
		int pos = index;
		for (int i = col; i > 0; i--) {
			if (traceMap.at<char>(pos, i) == 0) {		//向上走
				index -= 1;
			}
			else if (traceMap.at<char>(pos, i) == 2) {		//向下走
				index += 1;
			}
			Seam.at<float>(0, i - 1) = index;
			pos = index;
		}
	}
}
void ShowSeam(const Mat src,Mat Seam,int code) {
	Mat show(src.rows, src.cols, src.type());
	src.copyTo(show);
	if (code == 0) {	//用红线展示纵向Seam
		for (int i = 0; i < src.rows; i++) {
			int k = Seam.at<float>(i, 0);
			show.at<Vec3b>(i, k)[0] = 0;		//B
			show.at<Vec3b>(i, k)[1] = 0;		//G
			show.at<Vec3b>(i, k)[2] = 255;		//R
		}		
		imshow("ShowSeam",show);
//		waitKey(0);
	}
	else if (code == 1) {
		for (int i = 0; i < src.cols; i++) {
			int k = Seam.at<float>(0,i);
			show.at<Vec3b>(k, i)[0] = 0;
			show.at<Vec3b>(k, i)[1] = 0;
			show.at<Vec3b>(k, i)[2] = 255;
		}
		imshow("ShowSeam",show);
	}
}
void DeleteOneCol(Mat pre,Mat result,Mat seam) {
	for (int i = 0; i < pre.rows; i++) {
		int k = seam.at<float>(i, 0);
		for (int j = 0; j < k; j++) {			//seam左边的像素点不变
			result.at<Vec3b>(i, j)[0] = pre.at<Vec3b>(i, j)[0];
			result.at<Vec3b>(i, j)[1] = pre.at<Vec3b>(i, j)[1];
			result.at<Vec3b>(i, j)[2] = pre.at<Vec3b>(i, j)[2];
		}
		for (int j = k; j < pre.cols - 1; j++) {
			if (j == pre.cols - 1) continue;  //如果j=k=原图最后一列上的像素点，那么该行左边的像素点都已就位，直接进入下一行。
			result.at<Vec3b>(i, j)[0] = pre.at<Vec3b>(i, j + 1)[0];
			result.at<Vec3b>(i, j)[1] = pre.at<Vec3b>(i, j + 1)[1];
			result.at<Vec3b>(i, j)[2] = pre.at<Vec3b>(i, j + 1)[2];
		}
	}
}
void DeleteOneRow(Mat pre,Mat result,Mat seam) {
	for (int i = 0; i < pre.cols; i++) {
		int k = seam.at<float>(0, i);	
		for (int j = 0; j < k; j++) {	//seam上面的像素点不变
			result.at<Vec3b>(j, i)[0] = pre.at<Vec3b>(j, i)[0];
			result.at<Vec3b>(j, i)[1] = pre.at<Vec3b>(j, i)[1];
			result.at<Vec3b>(j, i)[2] = pre.at<Vec3b>(j, i)[2];
		}
		for (int j = k; j < pre.rows - 1; j++) {		//seam下方的点统一向上一个单位
			if (j == pre.rows - 1) continue;
			result.at<Vec3b>(j, i)[0] = pre.at<Vec3b>(j + 1, i)[0];
			result.at<Vec3b>(j, i)[1] = pre.at<Vec3b>(j + 1, i)[1];
			result.at<Vec3b>(j, i)[2] = pre.at<Vec3b>(j + 1, i)[2];
		}
	}
}

void run(Mat src, Mat result,int dirCode) {
	Mat gradientMap(src.rows, src.cols, CV_32F, Scalar(0));
	getEnergyMap(src, gradientMap);    //我们要的存每个像素点能量的矩阵，就是gradientMap.
	Mat energyAccumulation(src.rows, src.cols, CV_32F, Scalar(0));   //新建的能量累积矩阵
	Mat traceMap(src.rows, src.cols, CV_8UC1, Scalar(0));		 //新建的路径记录矩阵
	calculateEnergy(gradientMap, energyAccumulation, traceMap, dirCode); //注意 VERTICAL和HORIZONTAL都是指要寻找seam的类型
	//根据directionCode  新建一个装seam的Mat容器。 
	Mat Seam[2];
	Mat Vseam(src.rows,1,CV_32F,Scalar(0));
	Mat Hseam(1, src.cols, CV_32F, Scalar(0)); 
	Seam[0] = Vseam;
	Seam[1] = Hseam;
	FindSeam(energyAccumulation,traceMap,Seam[dirCode],dirCode);
//	ShowSeam(src,Seam[dirCode],dirCode);

	if (dirCode == 0) {  //删除了一列
		DeleteOneCol(src,result,Seam[dirCode]);
	}
	else if (dirCode == 1) {   //删除了一行 
		DeleteOneRow(src, result, Seam[dirCode]);
	}

}

int main(int argc, char* argv) {
	Mat original = imread("E:\\programming\\MATLAB\\pictures\\Beckham01.jpg");
	namedWindow("original");
	imshow("original",original);
	//waitKey(0);
	Mat temp;
	original.copyTo(temp);
	int linesToBeDeleted = 100;

	//横向变窄
	Mat finalImage(original.rows,original.cols-linesToBeDeleted,original.type());
	for (int i = 0; i < linesToBeDeleted; i++) {
		Mat tempResult(temp.rows, temp.cols - 1, temp.type());
		run(temp, tempResult, VERTICAL);		//注意 VERTICAL和HORIZONTAL都是指要寻找seam的类型
		temp = tempResult;
		if (i == linesToBeDeleted - 1)	tempResult.copyTo(finalImage);
	}

	//高度变矮
/*	Mat finalImage(original.rows - linesToBeDeleted, original.cols, original.type());
	for (int i = 0; i < linesToBeDeleted; i++) {
		Mat tempResult(temp.rows-1,temp.cols,temp.type());
		run(temp, tempResult, HORIZONTAL);
		temp = tempResult;
		if (i == linesToBeDeleted - 1) tempResult.copyTo(finalImage);
	}
*/
//	Mat finalImage(original.rows,original.cols - 1,original.type());		//纵向
//	Mat finalImage(original.rows - 1, original.cols, original.type());		//横向
//	run(temp,finalImage,VERTICAL);   //如果要显示seam ，将上面的for循环注释掉，并将run()中ShowSeam以下的部分注释掉
	imshow("final",finalImage);
	waitKey(0);
	
}

