#include <stdio.h>
#include <math.h>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <time.h>
#include"ronghe2.h"



#include <opencv2/opencv.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <cv.h>
#include <cxcore.h>
#include <highgui.h>
using namespace std;
using namespace cv;

class FindCountour
{
public:
	FindCountour(){}

IplImage *  translation(IplImage *src,int dix,int diy)//平移
{
	IplImage *pSrcImage = cvCreateImage(cvGetSize(src), src->depth, src->nChannels);

   for(int i=0;i<src->height;i++)
   {
	   for(int j=0;j<src->width;j++)
	   {
		   CvScalar color=cvGet2D(src,i,j);
		   cvSet2D(pSrcImage,(i+diy)%src->height,(j+dix)%src->width,color);
	   }
   }
   cvNamedWindow("dst");
   cvShowImage("dst",pSrcImage);
   //cvWaitKey(0);
   return pSrcImage;
   //cvReleaseImage(&pSrcImage);
   //cvReleaseImage(&src);
   cvDestroyWindow("dst");


}

IplImage* transform(IplImage *src,double scale)//缩放
{
	
	IplImage* dst=cvCreateImage(cvGetSize(src),IPL_DEPTH_8U, src->nChannels);
	cvCopy(src,dst,0);
	cvSetImageROI(dst,cvRect(121,45,137,208));
	cvNamedWindow("dst", CV_WINDOW_AUTOSIZE);
	cvShowImage("dst", dst); //dst 是剪裁后的图片
	cvWaitKey(0);	
	//double scale=0.5;
	 CvSize czSize;
	 czSize.width = src->width * scale;
	 czSize.height = src->height * scale; 
	IplImage *pSrcImage = cvCreateImage(czSize, src->depth, src->nChannels);//缩放后的照片
	cvResize(dst, pSrcImage, CV_INTER_AREA); //缩放图片
	cvNamedWindow("pSrcImage", CV_WINDOW_AUTOSIZE);
	cvShowImage("pSrcImage", pSrcImage); 
	cvWaitKey(0);	
	cvResetImageROI(dst);
	return  dst;
}
IplImage* rotateImage2(IplImage* src,int degree)  
{ 

	double angle = degree  * CV_PI / 180.; 
	double a = sin(angle), b = cos(angle); 
	int width=src->width, height=src->height;
	//旋转后的新图尺寸
	int width_rotate= int(height * fabs(a) + width * fabs(b));  
	int height_rotate=int(width * fabs(a) + height * fabs(b));  
	IplImage* img_rotate = cvCreateImage(cvSize(width_rotate, height_rotate), src->depth, src->nChannels);  
	//cvZero(img_rotate);  
	//保证原图可以任意角度旋转的最小尺寸
	int tempLength = sqrt((double)width * width + (double)height *height) + 10;  
	int tempX = (tempLength + 1) / 2 - width / 2;  
	int tempY = (tempLength + 1) / 2 - height / 2;  
	 //int tempX=0;
	 //int tempY=0;
	IplImage* temp = cvCreateImage(cvSize(tempLength, tempLength), src->depth, src->nChannels);  
	//cvZero(temp);  
	//将原图复制到临时图像tmp中心
	cvSetImageROI(temp, cvRect(tempX, tempY, width, height));  
	cvCopy(src, temp, NULL);  
	cvResetImageROI(temp);  
	//旋转数组map
	// [ m0  m1  m2 ] ===>  [ A11  A12   b1 ]
	// [ m3  m4  m5 ] ===>  [ A21  A22   b2 ]
	float m[6];  
	int w = temp->width;  
	int h = temp->height;  
	m[0] = b;  
	m[1] = a;  
	m[3] = -m[1];  
	m[4] = m[0];  
	// 将旋转中心移至图像中间  
	m[2] = w * 0.5f;  
	m[5] = h * 0.5f;  
	CvMat M = cvMat(2, 3, CV_32F, m);  
	cvGetQuadrangleSubPix(temp, img_rotate, &M);  
	cvReleaseImage(&temp);  
	return img_rotate;//旋转后的图片
} 


void rotateImage1(IplImage* img, IplImage *img_rotate,int degree)
{
	//旋转中心为图像中心
	CvPoint2D32f center;  
	center.x=float (img->width/2.0+0.5);
	center.y=float (img->height/2.0+0.5);
	//计算二维旋转的仿射变换矩阵
	float m[6];            
	CvMat M = cvMat( 2, 3, CV_32F, m );
	cv2DRotationMatrix( center, degree,1, &M);
	//变换图像，并用黑色填充其余值
	cvWarpAffine(img,img_rotate, &M,CV_INTER_LINEAR+CV_WARP_FILL_OUTLIERS,cvScalarAll(0) );
}



IplImage * originalpicture(IplImage *src,int Threshold,double scale,float degree,int dix,int diy)
  
{
 
	  IplImage *dst = cvCreateImage(cvGetSize(src), src->depth, src->nChannels);
	  cvCopy(src,dst,0);
	   cvNamedWindow("src", CV_WINDOW_AUTOSIZE);
	   cvShowImage("src", dst); //灰度图
	  cvWaitKey(0);
	  IplImage *gray =  cvCreateImage(cvGetSize(src), IPL_DEPTH_8U, 1);
	  cvCvtColor(dst, gray, CV_BGR2GRAY);
	  IplImage *binary = cvCreateImage(cvGetSize(gray), IPL_DEPTH_8U, 1); 
	 
	  cvThreshold(gray, binary, Threshold, 255, CV_THRESH_BINARY); 

	//cvNamedWindow("gray", CV_WINDOW_AUTOSIZE);
	//cvShowImage("gray", binary); //灰度图
	//cvWaitKey(0);

	
	CvMemStorage *pcvMStorage = cvCreateMemStorage();  
    CvSeq *pcvSeq = NULL;  
    cvFindContours(binary, pcvMStorage, &pcvSeq, sizeof(CvContour), CV_RETR_LIST    , CV_CHAIN_APPROX_SIMPLE, cvPoint(0, 0));//找出轮廓

	IplImage *pOutlineImage = cvCreateImage(cvGetSize(src), IPL_DEPTH_8U, 3); 
	//IplImage *pOutlineImage = cvCreateImage(cvGetSize(dst), IPL_DEPTH_8U, 3); 
	 cvDrawContours(pOutlineImage, pcvSeq, CV_RGB(255,0,0), CV_RGB(0,255,0), 1, 2);//画出轮廓

	//cvNamedWindow("grapcvSeqy", CV_WINDOW_AUTOSIZE);
	//cvShowImage("grapcvSeqy", pOutlineImage); //灰度图
	//cvWaitKey(0);
	IplImage *pOutImage = cvCreateImage(cvGetSize(src), IPL_DEPTH_8U, 3);
	// IplImage *pOutImage = cvCreateImage(cvGetSize(dst), IPL_DEPTH_8U, 3);
			
 //................................过滤出物体的轮廓.................................................//
		    double maxarea=0;  
           double max=0; 
     CvSeq*  tmppcvseq =pcvSeq;
	for(;tmppcvseq !=0; tmppcvseq = tmppcvseq->h_next)
	   {
		  // area=fabs(cvContourArea(pcvSeq,CV_WHOLE_SEQ));
		double area=fabs(cvContourArea(tmppcvseq));
		if(area >maxarea)
		{
			 maxarea=area;
		}
	}
	//cout<<"maxarea"<<maxarea<<endl;
	 

	CvSeq *cPrev = pcvSeq;
	CvSeq *cNext = NULL;
	bool first= true;
	for(CvSeq *c = pcvSeq;c!= NULL;c= cNext)
	{

	double	area1 = fabs(cvContourArea(c) );
		if(area1 <maxarea *0.01||area1==maxarea)
		{
			cNext = c->h_next;
			cvClearSeq(c);
			//cvClearMemStorage(c->storage);//回收内存
			continue;
		}//if
		else
		{
			if(first)
			pcvSeq = c;
			first = false;
			
	   cvDrawContours(pOutlineImage, c, CV_RGB(255,0,0), CV_RGB(0,255,0), 0,2,8);
		}
		cNext= c->h_next;
	}//for first_constours;


	cvNamedWindow("pOutlineImage1", CV_WINDOW_AUTOSIZE);
	cvShowImage("pOutlineImage1", pOutlineImage); //轮廓图
	cvWaitKey(0);



	CvSeq*  tmpseq =pcvSeq;
	for(;tmpseq !=0; tmpseq = tmpseq->h_next)
	{
		double Area=fabs(cvContourArea(tmpseq));
		//cout<<"Area===="<<Area<<endl; 
	}//判断是否只有一个area




	//。。。。。。。。。。。。。。。。。。。。。最小外接矩形。。。。。。。。。。。。。。。。。。。。。。。。。。。//
	IplImage *dst_img = cvCreateImage(cvGetSize(src), IPL_DEPTH_8U, 3); 
	//IplImage *dst_img = cvCreateImage(cvGetSize(dst), IPL_DEPTH_8U, 3); 
	CvBox2D rect = cvMinAreaRect2(pcvSeq);

	//cout<<"center.  "<<rect.center.x<<"  "<<rect.center.y<<endl;// 盒子的中心
	//cout<<"rect1.size.height=="<<rect.size.height<<"  "<<"rect1.size.width=="<<rect.size.width<<endl;//盒子的长和宽
	//cout<<"rect1.angle=="<<rect.angle<<endl;//注意夹角 angle 是水平轴逆时针旋转，与碰到的第一个边（不管是高还是宽）的夹角

//。。。。。。。。。。。。。。。。。。。。。。。。如果最小外接矩形是正的。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。//
	 if(rect.angle==0)
	  {
		  anglee=degree;
		CvPoint2D32f rect_pts0[4];
		cvBoxPoints(rect, rect_pts0);
		int npts = 4;
		CvPoint rect_pts[4], *pt = rect_pts;
	
		for (int rp=0; rp<4; rp++)
		{ 
			rect_pts[rp]= cvPointFrom32f(rect_pts0[rp]);
			//cout<<rp<<"  "<<".x"<<rect_pts[rp].x<<"  "<<".y"<<rect_pts[rp].y<<endl;//最小外接矩形的顶点坐标
		}
		cvPolyLine(dst_img, &pt, &npts, 1, 1, CV_RGB(0,255,0), 2);

		cvNamedWindow("dst_img", CV_WINDOW_AUTOSIZE);
		cvShowImage("dst_img", dst_img); //dst_img最小外接矩形的轮廓
		cvWaitKey(0);	
		
		cvSetImageROI(dst,cvRect(rect_pts[1].x,rect_pts[1].y,rect.size.width,rect.size.height));////剪裁


		//cvNamedWindow("dst", CV_WINDOW_AUTOSIZE);
		//cvShowImage("dst", dst); //dst 是剪裁后的图片
		//cvWaitKey(0);


		 CvSize czSize;
		 czSize.width = rect.size.width * scale;
		 czSize.height = rect.size.height * scale; 

		IplImage *pSrcImage = cvCreateImage(czSize, src->depth, src->nChannels);//缩放后的照片
		cvResize(dst, pSrcImage, CV_INTER_AREA); //pSrcImage是缩放后的图片
		Mat image2=pSrcImage;

		cvNamedWindow("suofang", CV_WINDOW_AUTOSIZE);
		cvShowImage("suofang", pSrcImage); //pSrcImage 是缩放后的图片
		cvWaitKey(0);	

		// CvSize czSize1;
		 czSize1.width =src->width *2;
		 czSize1.height = src->height *2 ;

		 // czSize1.width =dst->width *2;
		 //czSize1.height = dst->height *2 ; 

		 //cout<<"====="<<endl;
		 //cout<<"czSize.width==="<<czSize.width<<"   "<<"czSize.height "<<czSize.height <<endl;
		 //cout<<"czSize1.width ==="<<czSize1.width <<"  "<<"czSize1.height "<<czSize1.height <<endl;




		 Mat write1;
		 write1= Mat(czSize1,CV_8UC3);

		//Rect  rect0(50,50,czSize.width,czSize.height)	;
		 
		 Rect rect1(rect_pts[1].x,rect_pts[1].y,czSize.width,czSize.height);
		 image2.copyTo(write1(rect1));
		
		 imshow("image2",image2);
		 
		 IplImage *write=&IplImage(write1);

		IplImage *Image = cvCreateImage(czSize1, src->depth, src->nChannels);
		Image=rotateImage2(write,degree);//Image是旋转后的图片
			//Mat image1=Image;
	
		//。。。。。。。。。。。。deichuzuobiao 。。。。。。。。。。。。。。。。。。。。

		
	
		  IplImage *gray2 =  cvCreateImage(cvGetSize(Image), IPL_DEPTH_8U, 1);
		  cvCvtColor(Image, gray2, CV_BGR2GRAY);
		  IplImage *binary2 = cvCreateImage(cvGetSize(gray2), IPL_DEPTH_8U, 1); 
	 
		  cvThreshold(gray2, binary2, 200, 255, CV_THRESH_BINARY); 

  //      cvNamedWindow("binary", CV_WINDOW_AUTOSIZE);
		//cvShowImage("binary", binary2); //binary 是二值化后的图片
		//cvWaitKey(0);

		IplImage *out = cvCreateImage(cvGetSize(gray2), IPL_DEPTH_8U, 3);
		IplImage *out2 = cvCreateImage(cvGetSize(gray2), IPL_DEPTH_8U, 3);
		CvMemStorage *pcvMStorage2 = cvCreateMemStorage();  
		CvSeq *pcvSeq2 = NULL;  
		cvFindContours(binary2, pcvMStorage2, &pcvSeq2, sizeof(CvContour), CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE, cvPoint(0, 0)); 
		cvDrawContours(out, pcvSeq2, CV_RGB(255,0,0), CV_RGB(0,255,0), 1,2);

		//cvNamedWindow("out1", CV_WINDOW_AUTOSIZE);
		//cvShowImage("out1", out); //lunkuo
		//cvWaitKey(0);

		double a=czSize.height*czSize.width;
		double b=czSize1.height*czSize1.width*1.4;
		r=a/b;
		//cout<<"r==="<<r<<endl;
		
		    double maxarea2=0;  
     CvSeq*  tmp =pcvSeq2;
	for(;tmp !=0; tmp = tmp->h_next)
	   {
		  // area=fabs(cvContourArea(pcvSeq,CV_WHOLE_SEQ));
		double area3=fabs(cvContourArea(tmp));

		//if(area3!=0)
		//{cout<<"area3"<<area3<<endl;}

		if(area3 >maxarea2)
		{
			 maxarea2=area3;
		}
	}
	//	 cout<<"pcvSeq->"<<pcvSeq->elem_size<<endl;
	//cout<<"maxarea2"<<maxarea2<<endl;

	CvSeq *fPrev = pcvSeq2;
	CvSeq *fNext = NULL;
	bool firstt= true;
	for(CvSeq *f = pcvSeq2;f!= NULL;f= fNext)
	{

	double	area3 = fabs(cvContourArea(f) );
		if(area3<maxarea2 *0.004||area3==maxarea2)
		{
			fNext = f->h_next;
			cvClearSeq(f);
			//cvClearMemStorage(c->storage);//回收内存
			continue;
		}//if
		else
		{
			if(firstt)
			pcvSeq2 = f;
			firstt = false;
			
	   cvDrawContours(out2, f, CV_RGB(255,0,0), CV_RGB(0,255,0), 0,2,8);
		}
		fNext= f->h_next;
	}//for first_constours;


		//cout<<"pcvSeq1"<<pcvSeq2->total<<endl;
		CvSeq*  tmq =pcvSeq2;
	for(;tmq !=0; tmq = tmq->h_next)
	{
		double Area1=fabs(cvContourArea(tmq));
		//cout<<"Area1===="<<Area1<<endl; 
	}


		//cvNamedWindow("out2", CV_WINDOW_AUTOSIZE);
		//cvShowImage("out2", out2); //灰度图
		//cvWaitKey(0);



		IplImage *dst_img2 = cvCreateImage(cvGetSize(Image), IPL_DEPTH_8U, 3); 
		
		CvBox2D rect3 = cvMinAreaRect2(pcvSeq2);

		cout<<endl<<endl;
		//cout<<"center3.  "<<rect3.center.x<<"  "<<rect3.center.y<<endl;// 盒子的中心
		//cout<<"rect3.size.height=="<<rect3.size.height<<"  "<<"rect3.size.width=="<<rect3.size.width<<endl;//盒子的长和宽
		//cout<<"rect3.angle=="<<rect3.angle<<endl;//注意夹角 angle 是水平轴逆时针旋转，与碰到的第一个边（不管是高还是宽）的夹角


		CvPoint2D32f rect3_pts0[4];
		cvBoxPoints(rect3, rect3_pts0);
		int npt3 = 4;
		CvPoint rect3_pts[4], *ptt = rect3_pts;
	
		for (int rp=0; rp<4; rp++)
		{ 
			rect3_pts[rp]= cvPointFrom32f(rect3_pts0[rp]);
			//cout<<rp<<"  "<<".x"<<rect3_pts[rp].x<<"  "<<".y"<<rect3_pts[rp].y<<endl;//最小外接矩形的顶点坐标
		}
		cvPolyLine(dst_img2, &ptt, &npt3, 1, 1, CV_RGB(0,255,0), 2);

		cvNamedWindow("dst_img2", CV_WINDOW_AUTOSIZE);
		cvShowImage("dst_img2", dst_img2); //dst_img最小外接矩形的轮廓
		cvWaitKey(0);	

				IplImage* dst4=cvCreateImage(cvGetSize(Image),IPL_DEPTH_8U, Image->nChannels);
		cvCopy(Image,dst4,0);


		CvPoint p0,p1,p2,p3;
	
        int x0=rect3_pts[0].x;
		int y0=rect3_pts[0].y;
		int x1=rect3_pts[1].x;
		int y1=rect3_pts[1].y;
		int x2=rect3_pts[2].x;
		int y2=rect3_pts[2].y;
		int x3=rect3_pts[3].x;
		int y3=rect3_pts[3].y;

      
		p0.x=x1;
		p0.y=y2;
		p1.x=x3;
		p1.y=y2;
		p2.x=x1;
		p2.y=y0;
		p3.x=x3;
		p3.y=y0;

		//cout<<"p0.x"<<p0.x<<"p0.y"<<p0.y<<endl;
		//cout<<"p1.x"<<p1.x<<"p1.y"<<p1.y<<endl;
		//cout<<"p2.x"<<p2.x<<"p2.y"<<p2.y<<endl;
		//cout<<"p3.x"<<p3.x<<"p3.y"<<p3.y<<endl;

		cvLine( dst4, p0, p1,CV_RGB(0,255,0), 1, 8,0 );
		cvLine( dst4, p1, p3,CV_RGB(0,255,0), 1, 8,0 );
		cvLine( dst4, p3, p2,CV_RGB(0,255,0), 1, 8,0 );
		cvLine( dst4, p2, p0,CV_RGB(0,255,0), 1, 8,0 );
	  	

		cvNamedWindow("dst4", CV_WINDOW_AUTOSIZE);
		cvShowImage("dst4", dst4); //dst4是画了方框的图片
		cvWaitKey(0);
//..........................................................................................//

				Mat write2,image3;
		
	     IplImage* dst33=cvCreateImage(cvGetSize(Image), src->depth, src->nChannels);
		 IplImage *image11=cvCreateImage(cvGetSize(Image), src->depth, src->nChannels);

		 IplImage *image=cvCreateImage(cvGetSize(Image), src->depth, src->nChannels);

		 cvCopy(Image,dst33,0);

		 write2= Mat(czSize1,CV_8UC3);
		// CvSize cvSize;
		 cvSize4.height=abs(p3.x-p2.x);
		 cvSize4.width=abs(p0.y-p2.y);
		 cvSetImageROI(dst33,cvRect(p0.x,p0.y,cvSize4.height, cvSize4.width));

		 image3=dst33;
		 Rect rect11(10,10,cvSize4.height, cvSize4.width);
		 image3.copyTo(write2(rect11));

		 image11=&IplImage(write2);
		 image=rotateImage2(image11,0);




		 //cout<<"rect.height==="<<cvSize4.height<<"  "<<"cvSize.width==="<<cvSize4.width<<endl;
		 //cout<<"pic.height==="<<czSize1.height<<"  "<<"pic.width==="<< czSize1.width <<endl;
		// cout<<"能移动y距离==="<<czSize1.height-cvSize4.width-12<<"   "<<"能移动x距离"<<czSize1.width-cvSize4.height-12<<endl;

		 //.......................................................................//
		cout<<"p0.x"<<" "<<10<<"p0.y"<<" "<<10<<endl;
		cout<<"p1.x"<<p1.x-p0.x+10<<" "<<"p1.y"<<p1.y-p0.y+10<<endl;
		cout<<"p2.x"<<p2.x-p0.x+10<<" "<<"p2.y"<<p2.y-p0.y+10<<endl;
		cout<<"p3.x"<<p3.x-p0.x+10<<" "<<"p3.y"<<p3.y-p0.y+10<<endl;
		//if(point==1)
		{
		//CvPoint p00,p11,p22,p33;
	
		p00.x=0+10;
		p00.y=0+10;
		p11.x=p1.x-p0.x+10;
		p11.y=p1.y-p0.y+10;
		p22.x=p2.x-p0.x+10;
		p22.y=p2.y-p0.y+10;
		p33.x=p3.x-p0.x+10;
		p33.y=p3.y-p0.y+10;






		//cvLine( image, p00, p11,CV_RGB(0,255,0), 1, 8,0 );
		//cvLine( image, p11, p33,CV_RGB(0,255,0), 1, 8,0 );
		//cvLine( image, p33, p22,CV_RGB(0,255,0), 1, 8,0 );
		//cvLine( image, p22, p00,CV_RGB(0,255,0), 1, 8,0 );


		//cout<<endl<<endl;
		//cout<<"平移后坐标"<<endl<<endl;
		//cout<<"p0.x==="<<" "<<10+dix<<" "<<"p0.y==="<<10+diy<<endl;
		//cout<<"p1.x==="<<p1.x-p0.x+dix+10<<" "<<"p1.y==="<<p1.y-p0.y+diy+10<<endl;
		//cout<<"p2.x==="<<p2.x-p0.x+dix+10<<" "<<"p2.y==="<<p2.y-p0.y+diy+10<<endl;
		//cout<<"p3.x==="<<p3.x-p0.x+dix+10<<" "<<"p3.y==="<<p3.y-p0.y+diy+10<<endl;
		//cout<<"转动角度==="<< degree<<endl;



		}



		
		
		cvNamedWindow("Image", CV_WINDOW_AUTOSIZE);
		cvShowImage("Image", image); //Image 是剪裁，旋转，缩放的图片
		cvWaitKey(0);	

		return image;

//............................................................................................................................//



  }

	if(rect.angle!=0)//如果最小外接矩形是倾斜的
	{
		
			IplImage *zhuan = cvCreateImage(cvGetSize(dst_img),dst_img->depth, dst_img->nChannels);
			zhuan=rotateImage2(dst,-rect.angle);//旋转后的图片
			float anglee=-rect.angle-degree;

			
		cvNamedWindow("zhuan", CV_WINDOW_AUTOSIZE);
		cvShowImage("zhuan", zhuan); //binary 是二值化后的图片
		cvWaitKey(0);

		  IplImage* dst2=cvCreateImage(cvGetSize(zhuan),IPL_DEPTH_8U, src->nChannels);
		  cvCopy(zhuan,dst2,0);
		  IplImage *gray1 =  cvCreateImage(cvGetSize(zhuan), IPL_DEPTH_8U, 1);
		  cvCvtColor(zhuan, gray1, CV_BGR2GRAY);
		  IplImage *binary1 = cvCreateImage(cvGetSize(gray1), IPL_DEPTH_8U, 1); 
	 
		  cvThreshold(gray1, binary1, 200, 255, CV_THRESH_BINARY); 

  //      cvNamedWindow("binary", CV_WINDOW_AUTOSIZE);
		//cvShowImage("binary", binary1); //binary 是二值化后的图片
		//cvWaitKey(0);


		CvMemStorage *pcvMStorage1 = cvCreateMemStorage();  
		CvSeq *pcvSeq1 = NULL;  
		cvFindContours(binary1, pcvMStorage1, &pcvSeq1, sizeof(CvContour), CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE, cvPoint(0, 0));    

		//  cvNamedWindow("binary1", CV_WINDOW_AUTOSIZE);
		//cvShowImage("binary1", binary1); //binary 是二值化后的图片
		//cvWaitKey(0);

		//cvFindContours(binary,pcvMStorage,&pcvSeq,CV_RETR_EXTERNAL,CV_CHAIN_APPROX_NONE);
		IplImage *dst_img1 = cvCreateImage(cvGetSize(zhuan), IPL_DEPTH_8U, 3); 
		CvBox2D rect1 = cvMinAreaRect2(pcvSeq1);
		cout<<endl<<endl;
		//cout<<"center2.  "<<rect1.center.x<<"  "<<rect1.center.y<<endl;// 盒子的中心
		//cout<<"rect2.size.height=="<<rect1.size.height<<"  "<<"rect2.size.width=="<<rect1.size.width<<endl;//盒子的长和宽
		//cout<<"rect2.angle=="<<rect1.angle<<endl;//注意夹角 angle 是水平轴逆时针旋转，与碰到的第一个边（不管是高还是宽）的夹角

		CvPoint2D32f rect_pts1[4];
		cvBoxPoints(rect1, rect_pts1);
		int npt = 4;
		CvPoint rect_ptss[4], *pt1 = rect_ptss;

		for (int rp=0; rp<4; rp++)
		{ 
			rect_ptss[rp]= cvPointFrom32f(rect_pts1[rp]);
			//cout<<rp<<"  "<<".x"<<rect_ptss[rp].x<<"  "<<".y"<<rect_ptss[rp].y<<endl;//最小外接矩形的顶点坐标
		}

		cvPolyLine(dst_img1, &pt1, &npt, 1, 1, CV_RGB(0,255,0), 2);

		cvNamedWindow("dst_img1", CV_WINDOW_AUTOSIZE);
		cvShowImage("dst_img1", dst_img1); //dst_img1t 是最小外接矩形的轮廓图
		cvWaitKey(0);	  	


		// CvSize czSize;
		// czSize.width =rect.size.width;
		// czSize.height =rect.size.height;
		//IplImage* dst=cvCreateImage(czSize,IPL_DEPTH_8U, src->nChannels);
		 
		cvSetImageROI(dst2,cvRect(rect_ptss[1].x,rect_ptss[1].y,rect1.size.width,rect1.size.height));////剪裁

		//cvNamedWindow("dst2", CV_WINDOW_AUTOSIZE);
		//cvShowImage("dst2", dst2); //dst 是剪裁后的图片
		//cvWaitKey(0);

		 CvSize czSize;
		 czSize.width = rect.size.width * scale;
		 czSize.height = rect.size.height * scale; 
		
		IplImage *pSrcImage = cvCreateImage(czSize, src->depth, src->nChannels);//缩放后的照片
		cvResize(dst2, pSrcImage, CV_INTER_AREA); //缩放后的图片
		Mat image2=pSrcImage;

		cvNamedWindow("pSrcImage", CV_WINDOW_AUTOSIZE);
		cvShowImage("pSrcImage", pSrcImage); //dst 是剪裁后的图片
		cvWaitKey(0);	

		// CvSize czSize1;
		 czSize1.width = src->width  *2;
		 czSize1.height =src->height *2; 


		 //cout<<"czSize.width==="<<czSize.width<<"   "<<"czSize.height "<<czSize.height <<endl;
		 //cout<<"czSize1.width ==="<<czSize1.width <<"  "<<"czSize1.height "<<czSize1.height <<endl;




		 if(czSize.width>czSize1.width||czSize.height>czSize1.height)
		 {
			 cout<<"放大尺寸超出图片尺寸"<<endl;
		 }


		 Mat write1;
		 write1= Mat(czSize1,CV_8UC3);

		 Rect rect(rect_ptss[1].x,rect_ptss[1].y,czSize.width,czSize.height);
		 image2.copyTo(write1(rect));

		 IplImage *write=&IplImage(write1);

		IplImage *Image = cvCreateImage(czSize1, src->depth, src->nChannels);
		Image=rotateImage2(write,degree);//旋转后的图片

		//Mat image1=Image;
		//Mat image3=salt(image1,100000);

		//IplImage *image4=&IplImage(image3);

//...........................得出处理后的目标图片的坐标..........................................................................
		
		  IplImage *gray2 =  cvCreateImage(cvGetSize(Image), IPL_DEPTH_8U, 1);
		  cvCvtColor(Image, gray2, CV_BGR2GRAY);
		  IplImage *binary2 = cvCreateImage(cvGetSize(gray2), IPL_DEPTH_8U, 1); 
	 
		  cvThreshold(gray2, binary2, 200, 255, CV_THRESH_BINARY); 

  //      cvNamedWindow("binary", CV_WINDOW_AUTOSIZE);
		//cvShowImage("binary", binary2); //binary 是二值化后的图片
		//cvWaitKey(0);

		IplImage *out = cvCreateImage(cvGetSize(gray2), IPL_DEPTH_8U, 3);
		IplImage *out2 = cvCreateImage(cvGetSize(gray2), IPL_DEPTH_8U, 3);
		CvMemStorage *pcvMStorage2 = cvCreateMemStorage();  
		CvSeq *pcvSeq2 = NULL;  
		cvFindContours(binary2, pcvMStorage2, &pcvSeq2, sizeof(CvContour), CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE, cvPoint(0, 0)); 
		cvDrawContours(out, pcvSeq2, CV_RGB(255,0,0), CV_RGB(0,255,0), 1,2);

		//cvNamedWindow("out1", CV_WINDOW_AUTOSIZE);
		//cvShowImage("out1", out); //灰度图
		//cvWaitKey(0);

		    double maxarea2=0;  
     CvSeq*  tmp =pcvSeq2;
	for(;tmp !=0; tmp = tmp->h_next)
	   {
		  // area=fabs(cvContourArea(pcvSeq,CV_WHOLE_SEQ));
		double area3=fabs(cvContourArea(tmp));

		//if(area3!=0)
		//{cout<<"area3"<<area3<<endl;}

		if(area3 >maxarea2)
		{
			 maxarea2=area3;
		}
	}
	//	 cout<<"pcvSeq->"<<pcvSeq->elem_size<<endl;
	//cout<<"maxarea2"<<maxarea2<<endl;

		double a=czSize.height*czSize.width;
		double b=czSize1.height*czSize1.width*1.4;
		r=a/b;
		//cout<<"r==="<<r<<endl;

	CvSeq *fPrev = pcvSeq2;
	CvSeq *fNext = NULL;
	bool firstt= true;
	for(CvSeq *f = pcvSeq2;f!= NULL;f= fNext)
	{

	double	area3 = fabs(cvContourArea(f) );
		if(area3<maxarea2 *r||area3==maxarea2)
		{
			fNext = f->h_next;
			cvClearSeq(f);
			//cvClearMemStorage(c->storage);//回收内存
			continue;
		}//if
		else
		{
			if(firstt)
			pcvSeq2 = f;
			firstt = false;
			
	   cvDrawContours(out2, f, CV_RGB(255,0,0), CV_RGB(0,255,0), 0,2,8);
		}
		fNext= f->h_next;
	}//for first_constours;

		CvSeq*  tmq =pcvSeq2;
	for(;tmq !=0; tmq = tmq->h_next)
	{
		double Area1=fabs(cvContourArea(tmq));
		//cout<<"Area1===="<<Area1<<endl; 
	}


		cvNamedWindow("out2", CV_WINDOW_AUTOSIZE);
		cvShowImage("out2", out2); 
		cvWaitKey(0);



		IplImage *dst_img2 = cvCreateImage(cvGetSize(Image), IPL_DEPTH_8U, 3); 
		
		CvBox2D rect3 = cvMinAreaRect2(pcvSeq2);

		cout<<endl<<endl;
		//cout<<"center3.  "<<rect3.center.x<<"  "<<rect3.center.y<<endl;// 盒子的中心
		//cout<<"rect3.size.height=="<<rect3.size.width<<"  "<<"rect3.size.width=="<<rect3.size.height<<endl;//盒子的长和宽
		//cout<<"rect3.angle=="<<rect3.angle<<endl;//注意夹角 angle 是水平轴逆时针旋转，与碰到的第一个边（不管是高还是宽）的夹角


		CvPoint2D32f rect3_pts0[4];
		cvBoxPoints(rect3, rect3_pts0);
		int npt3 = 4;
		CvPoint rect3_pts[4], *ptt = rect3_pts;
	
		for (int rp=0; rp<4; rp++)
		{ 
			rect3_pts[rp]= cvPointFrom32f(rect3_pts0[rp]);
			//cout<<rp<<"  "<<".x"<<rect3_pts[rp].x<<"  "<<".y"<<rect3_pts[rp].y<<endl;//最小外接矩形的顶点坐标
		}
		cvPolyLine(dst_img2, &ptt, &npt3, 1, 1, CV_RGB(0,255,0), 2);

		cvNamedWindow("dst_img2", CV_WINDOW_AUTOSIZE);
		cvShowImage("dst_img2", dst_img2); //dst_img最小外接矩形的轮廓
		cvWaitKey(0);	


//..........................求出正矩形坐标.......................................//

		IplImage* dst4=cvCreateImage(cvGetSize(Image),IPL_DEPTH_8U, Image->nChannels);
		cvCopy(Image,dst4,0);

		CvPoint p0,p1,p2,p3;
	
        int x0=rect3_pts[0].x;
		int y0=rect3_pts[0].y;
		int x1=rect3_pts[1].x;
		int y1=rect3_pts[1].y;
		int x2=rect3_pts[2].x;
		int y2=rect3_pts[2].y;
		int x3=rect3_pts[3].x;
		int y3=rect3_pts[3].y;

      
		p0.x=x1;
		p0.y=y2;
		p1.x=x3;
		p1.y=y2;
		p2.x=x1;
		p2.y=y0;
		p3.x=x3;
		p3.y=y0;

		//cout<<"p0.x"<<p0.x<<"p0.y"<<p0.y<<endl;
		//cout<<"p1.x"<<p1.x<<"p1.y"<<p1.y<<endl;
		//cout<<"p2.x"<<p2.x<<"p2.y"<<p2.y<<endl;
		//cout<<"p3.x"<<p3.x<<"p3.y"<<p3.y<<endl;

		cvLine( dst4, p0, p1,CV_RGB(0,255,0), 1, 8,0 );
		cvLine( dst4, p1, p3,CV_RGB(0,255,0), 1, 8,0 );
		cvLine( dst4, p3, p2,CV_RGB(0,255,0), 1, 8,0 );
		cvLine( dst4, p2, p0,CV_RGB(0,255,0), 1, 8,0 );

		cvNamedWindow("dst4", CV_WINDOW_AUTOSIZE);
		cvShowImage("dst4", dst4); //Image 是剪裁，旋转，缩放的图片
		cvWaitKey(0);


//............................................................................................
	
		Mat write2,image3;
		
	     IplImage* dst33=cvCreateImage(cvGetSize(Image), src->depth, src->nChannels);
		 IplImage *image11=cvCreateImage(cvGetSize(Image), src->depth, src->nChannels);

		 IplImage *image=cvCreateImage(cvGetSize(Image), src->depth, src->nChannels);

		 cvCopy(Image,dst33,0);

		 write2= Mat(czSize1,CV_8UC3);
		// CvSize cvSize;
		cvSize4.height=abs(p3.x-p2.x);
		 cvSize4.width=abs(p0.y-p2.y);
		 cvSetImageROI(dst33,cvRect(p0.x,p0.y,cvSize4.height, cvSize4.width));

		 image3=dst33;
		 Rect rect11(10,10,cvSize4.height, cvSize4.width);
		 image3.copyTo(write2(rect11));

		 image11=&IplImage(write2);
		 image=rotateImage2(image11,0);


		 //cout<<"rect.height==="<<cvSize4.height<<"  "<<"cvSize.width==="<<cvSize4.width<<endl;
		 //cout<<"pic.height==="<<czSize1.height<<"  "<<"pic.width==="<< czSize1.width <<endl;
		 //cout<<"能移动y距离==="<<czSize1.height-cvSize4.width-12<<"   "<<"能移动x距离"<<czSize1.width-cvSize4.height-12<<endl;

		 //.......................................................................//
		cout<<"p0.x"<<" "<<10<<"p0.y"<<" "<<10<<endl;
		cout<<"p1.x"<<p1.x-p0.x+10<<" "<<"p1.y"<<p1.y-p0.y+10<<endl;
		cout<<"p2.x"<<p2.x-p0.x+10<<" "<<"p2.y"<<p2.y-p0.y+10<<endl;
		cout<<"p3.x"<<p3.x-p0.x+10<<" "<<"p3.y"<<p3.y-p0.y+10<<endl;
		//if(point==1)
		{
		//CvPoint p00,p11,p22,p33;
	
		p00.x=0+10;
		p00.y=0+10;
		p11.x=p1.x-p0.x+10;
		p11.y=p1.y-p0.y+10;
		p22.x=p2.x-p0.x+10;
		p22.y=p2.y-p0.y+10;
		p33.x=p3.x-p0.x+10;
		p33.y=p3.y-p0.y+10;






		//cvLine( image, p00, p11,CV_RGB(0,255,0), 1, 8,0 );
		//cvLine( image, p11, p33,CV_RGB(0,255,0), 1, 8,0 );
		//cvLine( image, p33, p22,CV_RGB(0,255,0), 1, 8,0 );
		//cvLine( image, p22, p00,CV_RGB(0,255,0), 1, 8,0 );


		//cout<<endl<<endl;
		//cout<<"平移后坐标"<<endl<<endl;
		//cout<<"p0.x==="<<" "<<10+dix<<" "<<"p0.y==="<<10+diy<<endl;
		//cout<<"p1.x==="<<p1.x-p0.x+dix+10<<" "<<"p1.y==="<<p1.y-p0.y+diy+10<<endl;
		//cout<<"p2.x==="<<p2.x-p0.x+dix+10<<" "<<"p2.y==="<<p2.y-p0.y+diy+10<<endl;
		//cout<<"p3.x==="<<p3.x-p0.x+dix+10<<" "<<"p3.y==="<<p3.y-p0.y+diy+10<<endl;
		//cout<<"转动角度==="<< degree<<endl;
		}
	
		
		cvNamedWindow("Image", CV_WINDOW_AUTOSIZE);
		cvShowImage("Image", image); //Image 是剪裁，旋转，缩放的图片
		cvWaitKey(0);	

		return image;

}
}


void getrectpoints( int dix,int diy)
{
		cout<<endl<<endl;
		cout<<"平移后坐标"<<endl<<endl;
		cout<<"p0.x==="<<" "<<p00.x+dix<<" "<<"p0.y==="<<p00.x+diy<<endl;
		cout<<"p1.x==="<<p11.x+dix<<" "<<"p1.y==="<<p11.y+diy<<endl;
		cout<<"p2.x==="<<p22.x+dix<<" "<<"p2.y==="<<p22.y+diy<<endl;
		cout<<"p3.x==="<<p33.x+dix<<" "<<"p3.y==="<<p33.y+diy<<endl;
		cout<<"转动角度==="<< anglee<<endl;

}



vector<int >getdistance()
{


		 cout<<"rect.height==="<<cvSize4.height<<"  "<<"cvSize.width==="<<cvSize4.width<<endl;
		 cout<<"pic.height==="<<czSize1.height<<"  "<<"pic.width==="<< czSize1.width <<endl;
		 cout<<"能移动y距离==="<<czSize1.height-cvSize4.width-12<<"   "<<"能移动x距离"<<czSize1.width-cvSize4.height-12<<endl;

		 vector<int>distance;
		 distance.clear();
		 distance.push_back(czSize1.width-cvSize4.height-12);
		  distance.push_back(czSize1.height-cvSize4.width-12);
		 return distance;
}



Mat salt(cv::Mat& image, int n)//椒盐噪声
{
    for(int k=0; k<n; k++){
        int i = rand()%image.cols;
        int j = rand()%image.rows;
        
        if(image.channels() == 1){
            image.at<uchar>(j,i) = 255;
        }else{
            image.at<cv::Vec3b>(j,i)[0] = 255;
            image.at<cv::Vec3b>(j,i)[1] = 255;
            image.at<cv::Vec3b>(j,i)[2] = 255;
        }
    }
	return image;
}


void backgroundpicture( IplImage *src,IplImage *dst1,int Threshold1,int Threshold2 ,double scale,float degree,double per,int dix,int diy,int noise )//Threshold=200
{
	 
	  IplImage *srcc=originalpicture(src,Threshold1,scale,degree,dix,diy);
	  getrectpoints(dix,diy);//输出rect的坐标


	   cvNamedWindow("srcc", CV_WINDOW_AUTOSIZE);
	  cvShowImage("srcc", srcc); //二值图
	  cvWaitKey(0);

	  Mat src1=srcc;
	  Mat dst=dst1;
	  IplImage *gray =  cvCreateImage(cvGetSize(srcc), IPL_DEPTH_8U, 1);
	  cvCvtColor(srcc, gray, CV_BGR2GRAY);
	  IplImage *binary = cvCreateImage(cvGetSize(gray), IPL_DEPTH_8U, 1); 
	 
	 
	  cvThreshold(gray, binary, Threshold2, 255, CV_THRESH_BINARY); 
	  //
	  //cvNamedWindow("binary", CV_WINDOW_AUTOSIZE);
	  //cvShowImage("binary", binary); //二值图
	  //cvWaitKey(0);

	CvMemStorage *pcvMStorage = cvCreateMemStorage();  
    CvSeq *pcvSeq = NULL;  
    cvFindContours(binary, pcvMStorage, &pcvSeq, sizeof(CvContour), CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE, cvPoint(0, 0));    
	IplImage *pOutlineImage = cvCreateImage(cvGetSize(srcc), IPL_DEPTH_8U, 3); 

	 cvDrawContours(pOutlineImage, pcvSeq, CV_RGB(255,0,0), CV_RGB(0,255,0), 1, 2); 

	//cvNamedWindow("pOutlineImage2", CV_WINDOW_AUTOSIZE);
	//cvShowImage("pOutlineImage2", pOutlineImage); //灰度图
	//cvWaitKey(0);

		double area=0;
	   double Area=0;
	   double acc=0;
	   CvSeq* maxContour =NULL;
	    CvSeq* Contour =NULL;
		
 
		    double maxarea=0;  
           double max=0; 
     CvSeq*  tmppcvseq =pcvSeq;
	for(;tmppcvseq !=0; tmppcvseq = tmppcvseq->h_next)
	   {
		  // area=fabs(cvContourArea(pcvSeq,CV_WHOLE_SEQ));
		double area=fabs(cvContourArea(tmppcvseq));
		if(area >maxarea)
		{
			 maxarea=area;
		}
	}

	//cout<<"maxarea"<<maxarea<<endl;

	 
	 int index = 0;
	int num_contours = 0;
	struct min_max_list *head = NULL;
	double area1= 0;
	CvSeq *cPrev = pcvSeq;
	CvSeq *cNext = NULL;
	bool first= true;
	for(CvSeq *c = pcvSeq;c!= NULL;c= cNext)
	{

		area1 = fabs(cvContourArea(c) );
		if(area1 <maxarea *r||area1==maxarea)
		{
			cNext = c->h_next;
			cvClearSeq(c);
			//cvClearMemStorage(c->storage);//回收内存
			continue;
		}//if
		else
		{
			if(first)
			pcvSeq = c;
			first = false;
			
	   cvDrawContours(pOutlineImage, c, CV_RGB(255,0,0), CV_RGB(0,255,0), 0,2,8);
		}
		cNext= c->h_next;
	}//for first_constours;


	//cvNamedWindow("pOutlineImage2", CV_WINDOW_AUTOSIZE);
	//cvShowImage("pOutlineImage2", pOutlineImage); //灰度图
	//cvWaitKey(0);

	CvSeq*  tmpseq =pcvSeq;
	for(;tmpseq !=0; tmpseq = tmpseq->h_next)
	{
		double Area=fabs(cvContourArea(tmpseq));
		//cout<<"Area===="<<Area<<endl; 
	}



 //  cvDrawContours(pOutlineImage, pcvSeq, CV_RGB(255,0,0), CV_RGB(0,255,0), 0, 2);  

	//cvNamedWindow("pstrWindowsOutLineTitle", CV_WINDOW_AUTOSIZE);
	//cvShowImage("pstrWindowsOutLineTitle", pOutlineImage); //pOutlineImage轮廓图
	//cvWaitKey(0);

	
	cv::Mat raw_dist( src1.size(), CV_32FC1); 	
	IplImage *raw_dist1=&IplImage(raw_dist);
	Mat mask;mask=Mat::zeros(src1.size(),CV_8UC1);
	IplImage *mask1=&IplImage(mask);



	 srand((unsigned)time(NULL));
	 IplImage *pSrcImage = cvCreateImage(cvGetSize(srcc), srcc->depth, srcc->nChannels);
	 Mat psrc=pSrcImage;
	 Mat new_image = Mat::zeros( psrc.size(), psrc.type() );
	IplImage *dstImg1=cvCreateImage(cvGetSize(srcc), IPL_DEPTH_8U, 3);
	IplImage *dstImg=&IplImage(dst);
	cvResize(dstImg, dstImg1, CV_INTER_AREA);


	
	//.........................	rgb	.........................
	for( int j = 0; j < src1.rows; j++ )
	{ 
		//
     for( int i = 0; i < src1.cols; i++ )
	 { 
		  
         raw_dist.at<float>(j,i) = cvPointPolygonTest( pcvSeq, Point2f(i,j), 0 );
		 
		  if(cvPointPolygonTest( pcvSeq, Point2f(i,j), 0 )>=0)
		  {

			   CvScalar s1=cvGet2D(dstImg,j,i);//背景图
			    CvScalar s3=s1;
			   CvScalar s2=cvGet2D(srcc,j,i);//目标图
			   //CvScalar color=cvGet2D(srcc,j,i);
			  double a=s1.val[0]*(1-per)+s2.val[0]*per;
			  double b=s1.val[1]*(1-per)+s2.val[1]*per;
			  double c=s1.val[2]*(1-per)+s2.val[2]*per;
			  s3.val[0]=a;
			  s3.val[1]=b;
			  s3.val[2]=c;
			 CvScalar color=s3;
			 if(noise==1)
			 {
			   if( s3.val[0]<150){color.val[0]=s3.val[0]+(rand()%101/100.0)*50;}
			   if(s3.val[0]>150){color.val[0]=s3.val[0]-(rand()%101/100.0)*50;}
			   if( s3.val[1]<150){color.val[1]=s3.val[1]+(rand()%101/100.0)*50;}
			   if(s3.val[1]>150){color.val[1]=s3.val[1]-(rand()%101/100.0)*50;}
			   if( s3.val[2]<150){color.val[2]=s3.val[2]+(rand()%101/100.0)*50;}
			   if(s3.val[2]>150){color.val[2]=s3.val[2]-(rand()%101/100.0)*50;}
			   //color.val[0]=s2.val[0]*(rand()%101/100.0);
			   //color.val[1]=s2.val[1]*(rand()%101/100.0);
			   //color.val[2]=s2.val[2]*(rand()%101/100.0);
			 }
			 
			 cvSet2D(pSrcImage,(j+diy)%srcc->height,(i+dix)%srcc->width,color);//移动目标图片		
			 	
			 for( int c = 0; c < 3; c++ )
            {
				if((j+diy)>srcc->height||(i+dix)>srcc->width)
				{cout<<"移动尺寸超出边界"<<endl;}
                //new_image.at<Vec3b>(j+diy,i+dix)[c] = saturate_cast<uchar>( alpha*( psrc.at<Vec3b>(j+diy,i+dix)[c] ) + beta );
				else{
				new_image.at<Vec3b>(j+diy,i+dix)[c] = saturate_cast<uchar>(( psrc.at<Vec3b>(j+diy,i+dix)[c] )  );
				}
                
			 }
			((uchar*)(mask1->imageData+mask1->widthStep	*(j+diy)))[i+dix]=255;
				//CvScalar s;
				//s=cvGet2D(srcc,i,j);

	 }


	 }

   }

  
	cvNamedWindow("pstrWindowsOutLineTitle", CV_WINDOW_AUTOSIZE);
	cvShowImage("pstrWindowsOutLineTitle", pSrcImage);//平移后的图片 
	
	imshow("mask",mask);//掩膜	
	imshow("new_image",new_image);
	 cvWaitKey(0); 

	//double RR=0,GG=0,BB=0;double averageRR;double averageGG;double averageBB;
 //   for(int i=0;i<rr.size();i++)
	//{
	//	RR+=rr[i]; GG+=gg[i];BB+=bb[i];
	//}
	//averageRR=RR/rr.size();
	//averageGG=GG/gg.size();
	//averageBB=BB/bb.size();
	//cout<<"averageBB=="<<averageRR<<endl;
//.........................koutu.........................................
	IplImage *image=&IplImage(new_image);
	//cvSmooth( image, image, CV_GAUSSIAN, 10,10 );,CV_BLUR,11,11
	//cvSmooth( image, image,CV_BLUR,11,11 );

	//cvSetImageROI(image, cvRect(20, 20, 64, 56));
 //   cvSetImageROI(dstImg1, cvRect(240, 210, 64, 56));
	// double beta = 1.0 - alpha;
	//cvAddWeighted(dstImg1,  0.5, image, 0.5, 0, image);//(src1, alpha, src2, beta, 0.0, dst);  
	
	Mat dstt=dstImg1;  
	new_image.copyTo(dstt,mask);//dstt 就是最终抠取的图片
	

//.......................................................................


	//cvNamedWindow("pstrWindowsOutLineTitle", CV_WINDOW_AUTOSIZE);
	//cvShowImage("pstrWindowsOutLineTitle", dstImg1);//平移后的图片 
	
	    imshow("dst",dstt);//最终图片
		cvSaveImage("save.png",dstImg1);//最终图片已经保存在文件夹里
        cvWaitKey(0); 
	    cvReleaseMemStorage(&pcvMStorage); 
	   // cvDestroyWindow("pstrWindowsSrcTitle");
		 cvReleaseImage(&srcc); 
		 cvReleaseImage(&gray);
		 cvReleaseImage(&binary);
		 cvReleaseImage(&pOutlineImage); 
}

double get_avg_gray(IplImage *img)
{
    IplImage *gray = cvCreateImage(cvGetSize(img),IPL_DEPTH_8U,1);
    cvCvtColor(img,gray,CV_RGB2GRAY);
    CvScalar scalar = cvAvg(gray);
    cvReleaseImage(&gray);
    return scalar.val[0];
}


void set_avg_gray(IplImage *img,IplImage *out,double avg_gray)
{
	double prev_avg_gray = get_avg_gray(img);
	cvConvertScale(img,out,avg_gray/prev_avg_gray);
}


public:
	 CvSize cvSize4;
     CvSize czSize1;
	 CvPoint p00,p11,p22,p33;//rect的坐标
	 float anglee;//转动角度
	float  r;//面积倍数
};
