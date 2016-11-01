#ifndef __TEST__

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

#define ALL_DIR      0
#define FOUR_DIR     1
#define TWO_DIR      2

typedef struct Bat_Config_struct
{
	double max_scale;
	double min_scale;
	int rotate_mode;
	bool with_noise;
	double min_percent;
}Bat_Config;

/****************************global variables*************************/

char foreground_file[80] = "E:\\VOCBat_4dir_nonoise\\1.png";
char background_path[80] = "E:\\VOCdevkit\\VOC2007\\JPEGImages\\";
char save_path[80] = "E:\\VOCBat_4dir_nonoise\\VOC2007\\JPEGImages\\";
char save_anno_file[80] = "E:\\VOCBat_4dir_nonoise\\VOC2007\\Annotations\\temp.txt";
int start_count = 1;
int total_iter = 9963;

Bat_Config config = 
{
	2.5,
	1.0,
	FOUR_DIR,
	false,
	0.75
};

/*********************************************************************/

string concat_file_name(char *path, int num)
{
	char num_str[10];
	string con;
	sprintf(num_str, "%06d", num);
	con = string(path) + string(num_str) + string(".jpg");
	return con;
}

void make_one_pic(FindCountour &f, IplImage *src_pic, IplImage *dst_pic, int crt_count, Bat_Config &c)
{	
	// function internal variables
	vector<int> distance;

	double scale;
	float degree;

	string file_name;
	double percent;
	int shift_x;
	int shift_y;
	int noise;
	
	try
	{
		// get scale ratio
		scale = ((double)(rand()%100)) * (config.max_scale - config.min_scale) / 100
				+ config.min_scale;
		// get rotation degree
		if(config.rotate_mode == ALL_DIR)
		{
			degree = (float)(rand()%360);
		}
		else if(config.rotate_mode == FOUR_DIR)
		{
			degree = (float)(rand()%4) * 90;
		}
		else
		{
			degree = (float)(rand()%2) * 180;
		}
		// get move distances
		f.originalpicture(src_pic, 150, scale, degree, 0, 0);
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
		// generate file name
		file_name = concat_file_name("", crt_count);
		// generate shift distance
		shift_x = rand()%(distance[0]);
		shift_y = rand()%(distance[1]);
		// generate noise
		if(config.with_noise == true)
		{
			noise = rand()%2;
		}
		else
		{
			noise = 0;
		}
		// generate mixture percent
		percent = ((double)(rand()%100)) * (1 - config.min_percent) / 100 + config.min_percent;
		// generate mixed picture
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

int main()
{
	FindCountour fc(save_path, save_anno_file, start_count);

	// read foreground picture 
	Mat origin_mat = imread(foreground_file, 1);
	IplImage *src = &IplImage(origin_mat);

	// random seed
	srand((unsigned int)time(NULL));

	for(int i = start_count; i < start_count + total_iter; i++)
	{
		// read background picture
		string file_name = concat_file_name(background_path, i);
		Mat back_mat = imread(file_name, 1);
		IplImage *dst = &IplImage(back_mat);
		// generate a mixed picture and save
		make_one_pic(fc, src, dst, i, config);
	}

	return 0;
}

#endif
