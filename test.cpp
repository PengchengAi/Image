#ifdef __TEST__

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
#include <opencv/cv.h>
#include <opencv/cxcore.h>
#include <opencv/highgui.h>
#include "findcountour-new.h"

using namespace cv;

using namespace std;


void extreme_test(FindCountour &f, IplImage * src_pic, IplImage * dst_pic)
{
	vector<int> distance;

	// scale 2, degree 359, put at 4 corners
	f.originalpicture(src_pic,150,2,359,0,0);
	distance = f.getdistance();
	f.backgroundpicture("extreme\\1.png",src_pic,dst_pic,150,200,2,359,0.5,0,0,0);

	f.originalpicture(src_pic,150,2,359,0,0);
	distance = f.getdistance();
	f.backgroundpicture("extreme\\2.png",src_pic,dst_pic,150,200,2,359,0.5,distance[0],0,0);

	f.originalpicture(src_pic,150,2,359,0,0);
	distance = f.getdistance();
	f.backgroundpicture("extreme\\3.png",src_pic,dst_pic,150,200,2,359,0.5,0,distance[1],0);

	f.originalpicture(src_pic,150,2,359,0,0);
	distance = f.getdistance();
	f.backgroundpicture("extreme\\4.png",src_pic,dst_pic,150,200,2,359,0.5,distance[0],distance[1],0);
	
	// scale 3, degree 45, put at 4 corners
	f.originalpicture(src_pic,150,3,45,0,0);
	distance = f.getdistance();
	f.backgroundpicture("extreme\\5.png",src_pic,dst_pic,150,200,3,45,0.5,0,0,0);

	f.originalpicture(src_pic,150,3,45,0,0);
	distance = f.getdistance();
	f.backgroundpicture("extreme\\6.png",src_pic,dst_pic,150,200,3,45,0.5,distance[0],0,0);

	f.originalpicture(src_pic,150,3,45,0,0);
	distance = f.getdistance();
	f.backgroundpicture("extreme\\7.png",src_pic,dst_pic,150,200,3,45,0.5,0,distance[1],0);

	f.originalpicture(src_pic,150,3,45,0,0);
	distance = f.getdistance();
	f.backgroundpicture("extreme\\8.png",src_pic,dst_pic,150,200,3,45,0.5,distance[0],distance[1],0);

	// scale 2, degree 0, put at 4 corners
	f.originalpicture(src_pic,150,2,0,0,0);
	distance = f.getdistance();
	f.backgroundpicture("extreme\\9.png",src_pic,dst_pic,150,200,2,0,0.5,0,0,0);

	f.originalpicture(src_pic,150,2,0,0,0);
	distance = f.getdistance();
	f.backgroundpicture("extreme\\10.png",src_pic,dst_pic,150,200,2,0,0.5,distance[0],0,0);

	f.originalpicture(src_pic,150,2,0,0,0);
	distance = f.getdistance();
	f.backgroundpicture("extreme\\11.png",src_pic,dst_pic,150,200,2,0,0.5,0,distance[1],0);

	f.originalpicture(src_pic,150,2,0,0,0);
	distance = f.getdistance();
	f.backgroundpicture("extreme\\12.png",src_pic,dst_pic,150,200,2,0,0.5,distance[0],distance[1],0);
	
	// scale 3, degree 90, put at 4 corners
	f.originalpicture(src_pic,150,3,90,0,0);
	distance = f.getdistance();
	f.backgroundpicture("extreme\\13.png",src_pic,dst_pic,150,200,3,90,0.5,0,0,0);

	f.originalpicture(src_pic,150,3,90,0,0);
	distance = f.getdistance();
	f.backgroundpicture("extreme\\14.png",src_pic,dst_pic,150,200,3,90,0.5,distance[0],0,0);

	f.originalpicture(src_pic,150,3,90,0,0);
	distance = f.getdistance();
	f.backgroundpicture("extreme\\15.png",src_pic,dst_pic,150,200,3,90,0.5,0,distance[1],0);

	f.originalpicture(src_pic,150,3,90,0,0);
	distance = f.getdistance();
	f.backgroundpicture("extreme\\16.png",src_pic,dst_pic,150,200,3,90,0.5,distance[0],distance[1],0);
	
	// scale 0.5, degree 225, put at 4 corners
	f.originalpicture(src_pic,150,0.5,225,0,0);
	distance = f.getdistance();
	f.backgroundpicture("extreme\\17.png",src_pic,dst_pic,150,200,0.5,225,0.5,0,0,0);

	f.originalpicture(src_pic,150,0.5,225,0,0);
	distance = f.getdistance();
	f.backgroundpicture("extreme\\18.png",src_pic,dst_pic,150,200,0.5,225,0.5,distance[0],0,0);

	f.originalpicture(src_pic,150,0.5,225,0,0);
	distance = f.getdistance();
	f.backgroundpicture("extreme\\19.png",src_pic,dst_pic,150,200,0.5,225,0.5,0,distance[1],0);

	f.originalpicture(src_pic,150,0.5,225,0,0);
	distance = f.getdistance();
	f.backgroundpicture("extreme\\20.png",src_pic,dst_pic,150,200,0.5,225,0.5,distance[0],distance[1],0);
}

void rand_test(FindCountour &f, IplImage *src_pic, IplImage *dst_pic, int count)
{
	vector<int> distance;

	double scale;
	float degree;

	string file_name;
	double percent;
	int shift_x;
	int shift_y;
	int noise;
	
	// random seed
	srand((unsigned int)time(NULL));

	// enter loop
	for(int i=0;i<count;i++)
	{
		try
		{
			scale = ((double)(rand()%100))*0.015 + 0.5;
			degree = (float)(rand()%360);
			f.originalpicture(src_pic,150,scale,degree,0,0);
			distance = f.getdistance();
			if(distance[0] < 1 || distance[1] < 1)
			{
				cout << "cannot move(max distance too small!)" << endl;
				throw exception();
			}
		}
		catch(exception &e)
		{
			cout << "originalpicture exception:" << endl;
			cout << "scale: " << scale << endl;
			cout << "degree: " << degree << endl;

			throw e;
		}

		try
		{
			char c[5];
			sprintf(c, "%d", i);
			file_name = string("random\\") + string(c) + string(".png");
			shift_x = rand()%(distance[0]);
			shift_y = rand()%(distance[1]);
			noise = rand()%2;
			percent = ((double)(rand()%100))*0.005 + 0.5;
			f.backgroundpicture(file_name.c_str(),src_pic,dst_pic,150,200,scale,degree,percent,shift_x,shift_y,noise);
		}
		catch(exception &e)
		{
			cout << "backgroundpicture exception:" << endl;
			cout << "scale: " << scale << endl;
			cout << "degree: " << degree << endl;
			cout << "percent: " << percent << endl;
			cout << "max_x: " << distance[0] << endl;
			cout << "max_y: " << distance[1] << endl;
			cout << "shift_x: " << shift_x << endl;
			cout << "shift_y: " << shift_y << endl;
			cout << "noise: " << noise << endl;

			throw e;
		}
	}
}

int main()
{
	Mat img11=imread("1.png",1);//原图
	IplImage *src=&IplImage(img11);
	Mat dst1=imread("pic.jpg",1);//背景图
	IplImage *dst=&IplImage(dst1);
	FindCountour fc("D:\\Pics\\");

	extreme_test(fc, src, dst);

	rand_test(fc, src, dst, 100);

	//fc.originalpicture(src,150,1.6,60,0,0);//此次执行必须使xy位移写为00

	//vector<int> distance;
	//distance.clear();
	//distance=fc.getdistance();
	//fc.backgroundpicture("save.png",src,dst,150,200,1.6,60,0.5,distance[0],distance[1],0);
	//fc.originalpicture(src,150,1,30);//originalpicture(IplImage *src,int Threshold,double scale,int degree)
	//fc.backgroundpicture(src,dst,150,200,1.6,60,0.5,393,271,1);
	//src--原图，dst--背景图，150--前景图片阈值， 200--背景图片阈值，1.6--缩放比例，
	//60--旋转角度, 0.5--图片颜色百分比，393--x方向平移距离，271--y方向平移距离	 1---是否注入噪声
	waitKey(0);
	return 0;
}

#endif