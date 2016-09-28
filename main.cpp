#include <stdio.h>
#include <math.h>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <time.h>


#include <opencv2/opencv.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <cv.h>
#include <cxcore.h>
#include <highgui.h>
#include"findcountour.h"

using namespace cv;

using namespace std;

int main()
{

	Mat img11=imread("1.png",1);//原图
	IplImage *src=&IplImage(img11);
	Mat dst1=imread("pic.jpg",1);//背景图
	IplImage *dst=&IplImage(dst1);
	FindCountour fc;

	fc.originalpicture(src,150,0.5,225,0,0);//此次执行必须使xy位移写为00
	vector<int>distance;
	distance.clear();
	distance=fc.getdistance();
	fc.backgroundpicture(src,dst,150,200,0.5,225,0.5,distance[0],distance[1],1);
	//	src--原图，dst--背景图，200--图片阈值，1--缩放比例，
	//60--旋转角度, 0.5--图片颜色百分比，200--x方向平移距离，200--y方向平移距离	 1---是否注入噪声   
	return 0;
}




