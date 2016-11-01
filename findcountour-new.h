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
using namespace std;
using namespace cv;

class FindCountour
{

private:
	string save_path_; 
	string output_file_;
	int main_count_;
	ofstream file_handler_;

public:
	FindCountour(char *save_path):
	save_path_(save_path), output_file_(""), main_count_(0)
	{}

	FindCountour(char *save_path, char *output_file, int start_count):
	save_path_(save_path), output_file_(output_file), main_count_(start_count)
	{
		file_handler_.open(output_file, ios::app);
	}

	~FindCountour()
	{
		if(file_handler_)
		{
			file_handler_.close();	
		}
	}

IplImage *  translation(IplImage *src,int dix,int diy)//ƽ��
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
   //cvNamedWindow("dst");
   //cvShowImage("dst",pSrcImage);
   //cvWaitKey(0);
   return pSrcImage;
   //cvReleaseImage(&pSrcImage);
   //cvReleaseImage(&src);
   cvDestroyWindow("dst");


}

IplImage* transform(IplImage *src,double scale)//����
{
	
	IplImage* dst=cvCreateImage(cvGetSize(src),IPL_DEPTH_8U, src->nChannels);
	cvCopy(src,dst,0);
	cvSetImageROI(dst,cvRect(121,45,137,208));
	cvNamedWindow("dst", CV_WINDOW_AUTOSIZE);
	cvShowImage("dst", dst); //dst �Ǽ��ú��ͼƬ
	cvWaitKey(0);	
	//double scale=0.5;
	 CvSize czSize;
	 czSize.width = src->width * scale;
	 czSize.height = src->height * scale; 
	IplImage *pSrcImage = cvCreateImage(czSize, src->depth, src->nChannels);//���ź����Ƭ
	cvResize(dst, pSrcImage, CV_INTER_AREA); //����ͼƬ
	cvNamedWindow("pSrcImage", CV_WINDOW_AUTOSIZE);
	cvShowImage("pSrcImage", pSrcImage); 
	cvWaitKey(0);	
	cvResetImageROI(dst);
	return  dst;
}


IplImage*	rotateImage2(IplImage* src,int degree)  
{ 

	double angle = degree  * CV_PI / 180.; 
	double a = sin(angle), b = cos(angle); 
	int width=src->width, height=src->height;
	//��ת�����ͼ�ߴ�
	int width_rotate= int(height * fabs(a) + width * fabs(b));  
	int height_rotate=int(width * fabs(a) + height * fabs(b));  
	IplImage* img_rotate = cvCreateImage(cvSize(width_rotate, height_rotate), src->depth, src->nChannels);  
	//cvZero(img_rotate);  
	//��֤ԭͼ��������Ƕ���ת����С�ߴ�
	int tempLength = sqrt((double)width * width + (double)height *height) + 10;  
	int tempX = (tempLength + 1) / 2 - width / 2;  
	int tempY = (tempLength + 1) / 2 - height / 2;  
	 //int tempX=0;
	 //int tempY=0;
	IplImage* temp = cvCreateImage(cvSize(tempLength, tempLength), src->depth, src->nChannels);  
	//cvZero(temp);  
	//��ԭͼ���Ƶ���ʱͼ��tmp����
	cvSetImageROI(temp, cvRect(tempX, tempY, width, height));  
	cvCopy(src, temp, NULL);  
	cvResetImageROI(temp);  
	//��ת����map
	// [ m0  m1  m2 ] ===>  [ A11  A12   b1 ]
	// [ m3  m4  m5 ] ===>  [ A21  A22   b2 ]
	float m[6];  
	int w = temp->width;  
	int h = temp->height;  
	m[0] = b;  
	m[1] = a;  
	m[3] = -m[1];  
	m[4] = m[0];  
	// ����ת��������ͼ���м�  
	m[2] = w * 0.5f;  
	m[5] = h * 0.5f;  
	CvMat M = cvMat(2, 3, CV_32F, m);  
	cvGetQuadrangleSubPix(temp, img_rotate, &M);

	cvReleaseImage(&temp); 

	return img_rotate;//��ת���ͼƬ
} 


void rotateImage1(IplImage* img, IplImage *img_rotate,int degree)
{
	//��ת����Ϊͼ������
	CvPoint2D32f center;  
	center.x=float (img->width/2.0+0.5);
	center.y=float (img->height/2.0+0.5);
	//�����ά��ת�ķ���任����
	float m[6];            
	CvMat M = cvMat( 2, 3, CV_32F, m );
	cv2DRotationMatrix( center, degree,1, &M);
	//�任ͼ�񣬲��ú�ɫ�������ֵ
	cvWarpAffine(img,img_rotate, &M,CV_INTER_LINEAR+CV_WARP_FILL_OUTLIERS,cvScalarAll(0) );
}



//
// IplImage *
void  originalpicture(IplImage *src,int Threshold,double scale,float degree,int dix,int diy)
  
{
 
	  IplImage *dst = cvCreateImage(cvGetSize(src), src->depth, src->nChannels);
	  cvCopy(src,dst,0);
	  // cvNamedWindow("src", CV_WINDOW_AUTOSIZE);
	  // cvShowImage("src", dst); //�Ҷ�ͼ
	  //cvWaitKey(0);
	  IplImage *gray =  cvCreateImage(cvGetSize(src), IPL_DEPTH_8U, 1);
	  cvCvtColor(dst, gray, CV_BGR2GRAY);
	  IplImage *binary = cvCreateImage(cvGetSize(gray), IPL_DEPTH_8U, 1); 
	 
	  cvThreshold(gray, binary, Threshold, 255, CV_THRESH_BINARY); 

	//cvNamedWindow("gray", CV_WINDOW_AUTOSIZE);
	//cvShowImage("gray", binary); //�Ҷ�ͼ
	//cvWaitKey(0);

	
	CvMemStorage *pcvMStorage = cvCreateMemStorage();  
    CvSeq *pcvSeq = NULL;  
    cvFindContours(binary, pcvMStorage, &pcvSeq, sizeof(CvContour), CV_RETR_LIST    , CV_CHAIN_APPROX_SIMPLE, cvPoint(0, 0));//�ҳ�����

	IplImage *pOutlineImage = cvCreateImage(cvGetSize(src), IPL_DEPTH_8U, 3); 
	//IplImage *pOutlineImage = cvCreateImage(cvGetSize(dst), IPL_DEPTH_8U, 3); 
	 cvDrawContours(pOutlineImage, pcvSeq, CV_RGB(255,0,0), CV_RGB(0,255,0), 1, 2);//��������

	//cvNamedWindow("grapcvSeqy", CV_WINDOW_AUTOSIZE);
	//cvShowImage("grapcvSeqy", pOutlineImage); //�Ҷ�ͼ
	//cvWaitKey(0);
	IplImage *pOutImage = cvCreateImage(cvGetSize(src), IPL_DEPTH_8U, 3);
	// IplImage *pOutImage = cvCreateImage(cvGetSize(dst), IPL_DEPTH_8U, 3);
			
 //................................���˳����������.................................................//
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
			//cvClearMemStorage(c->storage);//�����ڴ�
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


	//cvNamedWindow("pOutlineImage1", CV_WINDOW_AUTOSIZE);
	//cvShowImage("pOutlineImage1", pOutlineImage); //����ͼ
	//cvWaitKey(0);



	CvSeq*  tmpseq =pcvSeq;
	for(;tmpseq !=0; tmpseq = tmpseq->h_next)
	{
		double Area=fabs(cvContourArea(tmpseq));
		//cout<<"Area===="<<Area<<endl; 
	}//�ж��Ƿ�ֻ��һ��area




	//��������������������������������������������С��Ӿ��Ρ�����������������������������������������������������//
	IplImage *dst_img = cvCreateImage(cvGetSize(src), IPL_DEPTH_8U, 3); 
	//IplImage *dst_img = cvCreateImage(cvGetSize(dst), IPL_DEPTH_8U, 3); 
	CvBox2D rect = cvMinAreaRect2(pcvSeq);

	//cout<<"center.  "<<rect.center.x<<"  "<<rect.center.y<<endl;// ���ӵ�����
	//cout<<"rect1.size.height=="<<rect.size.height<<"  "<<"rect1.size.width=="<<rect.size.width<<endl;//���ӵĳ��Ϳ�
	//cout<<"rect1.angle=="<<rect.angle<<endl;//ע��н� angle ��ˮƽ����ʱ����ת���������ĵ�һ���ߣ������Ǹ߻��ǿ��ļн�

//�����������������������������������������������������С��Ӿ��������ġ�������������������������������������������������������������//
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
			//cout<<rp<<"  "<<".x"<<rect_pts[rp].x<<"  "<<".y"<<rect_pts[rp].y<<endl;//��С��Ӿ��εĶ�������
		}
		cvPolyLine(dst_img, &pt, &npts, 1, 1, CV_RGB(0,255,0), 2);

		//cvNamedWindow("dst_img", CV_WINDOW_AUTOSIZE);
		//cvShowImage("dst_img", dst_img); //dst_img��С��Ӿ��ε�����
		//cvWaitKey(0);	
		
		cvSetImageROI(dst,cvRect(rect_pts[1].x,rect_pts[1].y,rect.size.width,rect.size.height));////����


		//cvNamedWindow("dst", CV_WINDOW_AUTOSIZE);
		//cvShowImage("dst", dst); //dst �Ǽ��ú��ͼƬ
		//cvWaitKey(0);

		
		 CvSize czSize;
		 czSize.width = rect.size.width * scale;
		 czSize.height = rect.size.height * scale; 

		IplImage *pSrcImage = cvCreateImage(czSize, src->depth, src->nChannels);//���ź����Ƭ
		cvResize(dst, pSrcImage, CV_INTER_AREA); //pSrcImage�����ź��ͼƬ
		Mat image2=pSrcImage;

		//cvNamedWindow("suofang", CV_WINDOW_AUTOSIZE);
		//cvShowImage("suofang", pSrcImage); //pSrcImage �����ź��ͼƬ
		//cvWaitKey(0);	

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
		
		 //imshow("image2",image2);
		 
		 IplImage *write=&IplImage(write1);
//
//		IplImage *Image = cvCreateImage(czSize1, src->depth, src->nChannels);


		double angle = degree * CV_PI / 180.; 
		double a = sin(angle), b = cos(angle); 
		int width=write->width, height=write->height;
		//��ת�����ͼ�ߴ�
		int width_rotate= int(height * fabs(a) + width * fabs(b));  
		int height_rotate=int(width * fabs(a) + height * fabs(b));   
		IplImage* Image = cvCreateImage(cvSize(width_rotate, height_rotate), write->depth, write->nChannels);  
		//cvZero(img_rotate);  
		//��֤ԭͼ��������Ƕ���ת����С�ߴ�
		int tempLength = sqrt((double)width * width + (double)height *height) + 10;  
		int tempX = (tempLength + 1) / 2 - width / 2;  
		int tempY = (tempLength + 1) / 2 - height / 2;  
		 //int tempX=0;
		 //int tempY=0;
		IplImage* temp = cvCreateImage(cvSize(tempLength, tempLength), write->depth, write->nChannels);  
		//cvZero(temp);  
		//��ԭͼ���Ƶ���ʱͼ��tmp����
		cvSetImageROI(temp, cvRect(tempX, tempY, width, height));  
		cvCopy(write, temp, NULL);  
		cvResetImageROI(temp);  
		//��ת����map
		// [ m0  m1  m2 ] ===>  [ A11  A12   b1 ]
		// [ m3  m4  m5 ] ===>  [ A21  A22   b2 ]
		float m[6];  
		int w = temp->width;  
		int h = temp->height;  
		m[0] = b;  
		m[1] = a;  
		m[3] = -m[1];  
		m[4] = m[0];  
		// ����ת��������ͼ���м�  
		m[2] = w * 0.5f;  
		m[5] = h * 0.5f;  
		CvMat M = cvMat(2, 3, CV_32F, m);  
		cvGetQuadrangleSubPix(temp, Image, &M);
		cvReleaseImage(&temp);

//		  
//		Image=rotateImage2(write,degree);//Image����ת���ͼƬ

		//		cvNamedWindow("Image", CV_WINDOW_AUTOSIZE);
		//cvShowImage("Image", Image); //dst �Ǽ��ú��ͼƬ
		//cvWaitKey(0);




//	
//		//������������������������deichuzuobiao ����������������������������������������

		
	
		  IplImage *gray2 =  cvCreateImage(cvGetSize(Image), IPL_DEPTH_8U, 1);
		  cvCvtColor(Image, gray2, CV_BGR2GRAY);
		  IplImage *binary2 = cvCreateImage(cvGetSize(gray2), IPL_DEPTH_8U, 1); 
	 
		  cvThreshold(gray2, binary2, 200, 255, CV_THRESH_BINARY); 

  //      cvNamedWindow("binary", CV_WINDOW_AUTOSIZE);
		//cvShowImage("binary", binary2); //binary �Ƕ�ֵ�����ͼƬ
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

		double aa=czSize.height*czSize.width;
		double bb=czSize1.height*czSize1.width*3;
		 r=aa/bb;
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
		if(area3<maxarea2 *r||area3==maxarea2)
		{
			fNext = f->h_next;
			cvClearSeq(f);
			//cvClearMemStorage(c->storage);//�����ڴ�
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
		//cvShowImage("out2", out2); //�Ҷ�ͼ
		//cvWaitKey(0);



		IplImage *dst_img2 = cvCreateImage(cvGetSize(Image), IPL_DEPTH_8U, 3); 
		
		CvBox2D rect3 = cvMinAreaRect2(pcvSeq2);

		cout<<endl<<endl;
		//cout<<"center3.  "<<rect3.center.x<<"  "<<rect3.center.y<<endl;// ���ӵ�����
		//cout<<"rect3.size.height=="<<rect3.size.height<<"  "<<"rect3.size.width=="<<rect3.size.width<<endl;//���ӵĳ��Ϳ�
		//cout<<"rect3.angle=="<<rect3.angle<<endl;//ע��н� angle ��ˮƽ����ʱ����ת���������ĵ�һ���ߣ������Ǹ߻��ǿ��ļн�


		CvPoint2D32f rect3_pts0[4];
		cvBoxPoints(rect3, rect3_pts0);
		int npt3 = 4;
		CvPoint rect3_pts[4], *ptt = rect3_pts;
	
		for (int rp=0; rp<4; rp++)
		{ 
			rect3_pts[rp]= cvPointFrom32f(rect3_pts0[rp]);
			//cout<<rp<<"  "<<".x"<<rect3_pts[rp].x<<"  "<<".y"<<rect3_pts[rp].y<<endl;//��С��Ӿ��εĶ�������
		}
		cvPolyLine(dst_img2, &ptt, &npt3, 1, 1, CV_RGB(0,255,0), 2);

		//cvNamedWindow("dst_img2", CV_WINDOW_AUTOSIZE);
		//cvShowImage("dst_img2", dst_img2); //dst_img��С��Ӿ��ε�����
		//cvWaitKey(0);	

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
	  	

		//cvNamedWindow("dst4", CV_WINDOW_AUTOSIZE);
		//cvShowImage("dst4", dst4); //dst4�ǻ��˷����ͼƬ
		//cvWaitKey(0);
//..........................................................................................//

				Mat write2,image3;
		
	     IplImage* dst33=cvCreateImage(cvGetSize(Image), src->depth, src->nChannels);
		//IplImage *image11=cvCreateImage(cvGetSize(Image), src->depth, src->nChannels);

		// IplImage *image=cvCreateImage(cvGetSize(Image), src->depth, src->nChannels);

		 cvCopy(Image,dst33,0);

		 write2= Mat(czSize1,CV_8UC3);
		// CvSize cvSize;
		 cvSize4.height=abs(p3.x-p2.x);
		 cvSize4.width=abs(p0.y-p2.y);

		 cvSetImageROI(dst33,cvRect(p0.x,p0.y,cvSize4.height, cvSize4.width));

		 image3=dst33;
		 Rect rect11(10,10,cvSize4.height, cvSize4.width);

		 image3.copyTo(write2(rect11));

		IplImage * image11=&IplImage(write2);
//
//
////.............................................................
		double angle1 =0 * CV_PI / 180.; 
		double a1 = sin(angle1), b1 = cos(angle1); 
		int width1=image11->width, height1=image11->height;
		//��ת�����ͼ�ߴ�
		int width_rotate1= int(height1 * fabs(a1) + width1 * fabs(b1));  
		int height_rotate1=int(width1 * fabs(a1) + height1 * fabs(b1));   
		//IplImage* image = cvCreateImage(cvSize(width_rotate1, height_rotate1), image11->depth, image11->nChannels);
		imagee= cvCreateImage(cvSize(width_rotate1, height_rotate1), image11->depth, image11->nChannels);
		//cvZero(img_rotate);  
		//��֤ԭͼ��������Ƕ���ת����С�ߴ�
		int tempLength1 = sqrt((double)width1 * width1 + (double)height1 *height1) + 10;  
		int tempX1 = (tempLength1 + 1) / 2 - width1 / 2;  
		int tempY1 = (tempLength1 + 1) / 2 - height1 / 2;  
		IplImage* temp1 = cvCreateImage(cvSize(tempLength1, tempLength1), image11->depth, image11->nChannels);  
		//cvZero(temp);  
		//��ԭͼ���Ƶ���ʱͼ��tmp����
		cvSetImageROI(temp1, cvRect(tempX1, tempY1, width1, height1));  
		cvCopy(image11, temp1, NULL);  
		cvResetImageROI(temp1);  
		//��ת����map
		 //[ m0  m1  m2 ] ===>  [ A11  A12   b1 ]
		 //[ m3  m4  m5 ] ===>  [ A21  A22   b2 ]
		float m1[6];  
		int w1 = temp1->width;  
		int h1 = temp1->height;  
		m1[0] = b1;  
		m1[1] = a1;  
		m1[3] = -m1[1];  
		m1[4] = m1[0];  
		// ����ת��������ͼ���м�  
		m1[2] = w1 * 0.5f;  
		m1[5] = h1 * 0.5f;  
		CvMat M1 = cvMat(2, 3, CV_32F, m1);  
		cvGetQuadrangleSubPix(temp1, imagee, &M1);
		cvReleaseImage(&temp1);   
		//.....................................


		//image=rotateImage2(image11,0);

	//		cvNamedWindow("Image", CV_WINDOW_AUTOSIZE);
	//cvShowImage("Image", image); //Image �Ǽ��ã���ת�����ŵ�ͼƬ
	//cvWaitKey(0);




		 //cout<<"rect.height==="<<cvSize4.height<<"  "<<"cvSize.width==="<<cvSize4.width<<endl;
		 //cout<<"pic.height==="<<czSize1.height<<"  "<<"pic.width==="<< czSize1.width <<endl;
		// cout<<"���ƶ�y����==="<<czSize1.height-cvSize4.width-12<<"   "<<"���ƶ�x����"<<czSize1.width-cvSize4.height-12<<endl;

		 //.......................................................................//
		//cout<<"p0.x"<<" "<<10<<"p0.y"<<" "<<10<<endl;
		//cout<<"p1.x"<<p1.x-p0.x+10<<" "<<"p1.y"<<p1.y-p0.y+10<<endl;
		//cout<<"p2.x"<<p2.x-p0.x+10<<" "<<"p2.y"<<p2.y-p0.y+10<<endl;
		//cout<<"p3.x"<<p3.x-p0.x+10<<" "<<"p3.y"<<p3.y-p0.y+10<<endl;

		{
	
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
		//cout<<"ƽ�ƺ�����"<<endl<<endl;
		//cout<<"p0.x==="<<" "<<10+dix<<" "<<"p0.y==="<<10+diy<<endl;
		//cout<<"p1.x==="<<p1.x-p0.x+dix+10<<" "<<"p1.y==="<<p1.y-p0.y+diy+10<<endl;
		//cout<<"p2.x==="<<p2.x-p0.x+dix+10<<" "<<"p2.y==="<<p2.y-p0.y+diy+10<<endl;
		//cout<<"p3.x==="<<p3.x-p0.x+dix+10<<" "<<"p3.y==="<<p3.y-p0.y+diy+10<<endl;
		//cout<<"ת���Ƕ�==="<< degree<<endl;



		}	
		
		//cvNamedWindow("Image", CV_WINDOW_AUTOSIZE);
		//cvShowImage("Image", image); //Image �Ǽ��ã���ת�����ŵ�ͼƬ
		//cvWaitKey(0);


		cvResetImageROI(dst); 
		cvResetImageROI(dst33); 
		 cvReleaseImage(&dst); 
		 cvReleaseImage(&gray);
		 cvReleaseImage(&binary);
		 cvReleaseImage(&pOutlineImage); 
		 cvReleaseImage(&pOutImage);
		 cvReleaseImage(&dst_img);
		cvReleaseImage(&pSrcImage);
		 cvReleaseImage(&Image);
         cvReleaseImage(&gray2);
		 cvReleaseImage(&binary2);
		 cvReleaseImage(&out);
		 cvReleaseImage(&out2);
		  cvReleaseMemStorage(&pcvMStorage); 
		  cvReleaseMemStorage(&pcvMStorage2); 
		 cvReleaseImage(&dst_img2);
		 cvReleaseImage(&dst4);
		 cvReleaseImage(&dst33);

		//return image;

//............................................................................................................................//



  }

	if(rect.angle!=0)//�����С��Ӿ�������б��
	{
		double angle = -rect.angle  * CV_PI / 180.; 
		double a = sin(angle), b = cos(angle); 
		int width=dst->width, height=dst->height;
		//��ת�����ͼ�ߴ�
		int width_rotate= int(height * fabs(a) + width * fabs(b));  
		int height_rotate=int(width * fabs(a) + height * fabs(b));   
		IplImage* zhuan = cvCreateImage(cvSize(width_rotate, height_rotate), dst->depth, dst->nChannels);  
		//cvZero(img_rotate);  
		//��֤ԭͼ��������Ƕ���ת����С�ߴ�
		int tempLength = sqrt((double)width * width + (double)height *height) + 10;  
		int tempX = (tempLength + 1) / 2 - width / 2;  
		int tempY = (tempLength + 1) / 2 - height / 2;  
		 //int tempX=0;
		 //int tempY=0;
		IplImage* temp = cvCreateImage(cvSize(tempLength, tempLength), dst->depth, dst->nChannels);  
		//cvZero(temp);  
		//��ԭͼ���Ƶ���ʱͼ��tmp����
		cvSetImageROI(temp, cvRect(tempX, tempY, width, height));  
		cvCopy(dst, temp, NULL);  
		cvResetImageROI(temp);  
		//��ת����map
		// [ m0  m1  m2 ] ===>  [ A11  A12   b1 ]
		// [ m3  m4  m5 ] ===>  [ A21  A22   b2 ]
		float m[6];  
		int w = temp->width;  
		int h = temp->height;  
		m[0] = b;  
		m[1] = a;  
		m[3] = -m[1];  
		m[4] = m[0];  
		// ����ת��������ͼ���м�  
		m[2] = w * 0.5f;  
		m[5] = h * 0.5f;  
		CvMat M = cvMat(2, 3, CV_32F, m);  
		cvGetQuadrangleSubPix(temp, zhuan, &M);
		cvReleaseImage(&temp); 
			//IplImage *zhuan = cvCreateImage(cvGetSize(dst_img),dst_img->depth, dst_img->nChannels);

		//IplImage *	img_rotate = cvCreateImage(cvSize(width_rotate,height_rotate),dst->depth, dst->nChannels);
			//zhuan=rotateImage2(dst,-rect.angle);//��ת���ͼƬ
			//zhuan=rotateImage2(dst,img_rotate,-rect.angle);
    
		  anglee=-rect.angle-degree;


			
		//cvNamedWindow("zhuan", CV_WINDOW_AUTOSIZE);
		//cvShowImage("zhuan", img_rotate); //binary �Ƕ�ֵ�����ͼƬ
		//cvWaitKey(0); 
		  IplImage* dst2=cvCreateImage(cvGetSize(zhuan),IPL_DEPTH_8U, src->nChannels);
		  cvCopy(zhuan,dst2,0);
		  IplImage *gray1 =  cvCreateImage(cvGetSize(zhuan), IPL_DEPTH_8U, 1);
		  cvCvtColor(zhuan, gray1, CV_BGR2GRAY);
		  IplImage *binary1 = cvCreateImage(cvGetSize(gray1), IPL_DEPTH_8U, 1); 
	 
		  cvThreshold(gray1, binary1, 200, 255, CV_THRESH_BINARY); 

  ////      cvNamedWindow("binary", CV_WINDOW_AUTOSIZE);
		////cvShowImage("binary", binary1); //binary �Ƕ�ֵ�����ͼƬ
		////cvWaitKey(0);


		CvMemStorage *pcvMStorage1 = cvCreateMemStorage();  
		CvSeq *pcvSeq1 = NULL;  
		cvFindContours(binary1, pcvMStorage1, &pcvSeq1, sizeof(CvContour), CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE, cvPoint(0, 0));    

		//  cvNamedWindow("binary1", CV_WINDOW_AUTOSIZE);
		//cvShowImage("binary1", binary1); //binary �Ƕ�ֵ�����ͼƬ
		//cvWaitKey(0);

		//cvFindContours(binary,pcvMStorage,&pcvSeq,CV_RETR_EXTERNAL,CV_CHAIN_APPROX_NONE);
		IplImage *dst_img1 = cvCreateImage(cvGetSize(zhuan), IPL_DEPTH_8U, 3); 
		CvBox2D rect1 = cvMinAreaRect2(pcvSeq1);
		cout<<endl<<endl;
		//cout<<"center2.  "<<rect1.center.x<<"  "<<rect1.center.y<<endl;// ���ӵ�����
		//cout<<"rect2.size.height=="<<rect1.size.height<<"  "<<"rect2.size.width=="<<rect1.size.width<<endl;//���ӵĳ��Ϳ�
		//cout<<"rect2.angle=="<<rect1.angle<<endl;//ע��н� angle ��ˮƽ����ʱ����ת���������ĵ�һ���ߣ������Ǹ߻��ǿ��ļн�

		CvPoint2D32f rect_pts1[4];
		cvBoxPoints(rect1, rect_pts1);
		int npt = 4;
		CvPoint rect_ptss[4], *pt1 = rect_ptss;

		for (int rp=0; rp<4; rp++)
		{ 
			rect_ptss[rp]= cvPointFrom32f(rect_pts1[rp]);
			//cout<<rp<<"  "<<".x"<<rect_ptss[rp].x<<"  "<<".y"<<rect_ptss[rp].y<<endl;//��С��Ӿ��εĶ�������
		}

		cvPolyLine(dst_img1, &pt1, &npt, 1, 1, CV_RGB(0,255,0), 2);

		//cvNamedWindow("dst_img1", CV_WINDOW_AUTOSIZE);
		//cvShowImage("dst_img1", dst_img1); //dst_img1t ����С��Ӿ��ε�����ͼ
		//cvWaitKey(0);	  	


		// CvSize czSize;
		// czSize.width =rect.size.width;
		// czSize.height =rect.size.height;
		//IplImage* dst=cvCreateImage(czSize,IPL_DEPTH_8U, src->nChannels);
		 
		cvSetImageROI(dst2,cvRect(rect_ptss[1].x,rect_ptss[1].y,rect1.size.width,rect1.size.height));////����

		//cvNamedWindow("dst2", CV_WINDOW_AUTOSIZE);
		//cvShowImage("dst2", dst2); //dst �Ǽ��ú��ͼƬ
		//cvWaitKey(0);

		 CvSize czSize;
		 czSize.width = rect.size.width * scale;
		 czSize.height = rect.size.height * scale; 
		
		IplImage *pSrcImage = cvCreateImage(czSize, src->depth, src->nChannels);//���ź����Ƭ
		cvResize(dst2, pSrcImage, CV_INTER_AREA); //���ź��ͼƬ
		Mat image2=pSrcImage;

		//cvNamedWindow("pSrcImage", CV_WINDOW_AUTOSIZE);
		//cvShowImage("pSrcImage", pSrcImage); //dst �Ǽ��ú��ͼƬ
		//cvWaitKey(0);	

		// CvSize czSize1;
		 czSize1.width = src->width  *2;
		 czSize1.height =src->height *2; 


		 //cout<<"czSize.width==="<<czSize.width<<"   "<<"czSize.height "<<czSize.height <<endl;
		 //cout<<"czSize1.width ==="<<czSize1.width <<"  "<<"czSize1.height "<<czSize1.height <<endl;




		 if(czSize.width>czSize1.width||czSize.height>czSize1.height)
		 {
			 cout<<"�Ŵ�ߴ糬��ͼƬ�ߴ�"<<endl;
		 }


		 Mat write1;
		 write1= Mat(czSize1,CV_8UC3);

		 Rect rect(rect_ptss[1].x,rect_ptss[1].y,czSize.width,czSize.height);
		 image2.copyTo(write1(rect));

		 IplImage *write=&IplImage(write1);

		//IplImage *Image = cvCreateImage(czSize1, src->depth, src->nChannels);

//........................................................................................

		double angle1 =degree * CV_PI / 180.; 
		double a1 = sin(angle), b1 = cos(angle); 
		int width1=write->width, height1=write->height;
		//��ת�����ͼ�ߴ�
		int width_rotate1= int(height1 * fabs(a1) + width1 * fabs(b1));  
		int height_rotate1=int(width1 * fabs(a1) + height1 * fabs(b1));   
		IplImage* Image = cvCreateImage(cvSize(width_rotate1, height_rotate1), write->depth, write->nChannels);  
		//cvZero(img_rotate);  
		//��֤ԭͼ��������Ƕ���ת����С�ߴ�
		int tempLength1 = sqrt((double)width1 * width1 + (double)height1 *height1) + 10;  
		int tempX1 = (tempLength1 + 1) / 2 - width1 / 2;  
		int tempY1 = (tempLength1 + 1) / 2 - height1 / 2;  
		 //int tempX=0;
		 //int tempY=0;
		IplImage* temp1 = cvCreateImage(cvSize(tempLength1, tempLength1), write->depth, write->nChannels);  
		//cvZero(temp);  
		//��ԭͼ���Ƶ���ʱͼ��tmp����
		cvSetImageROI(temp1, cvRect(tempX1, tempY1, width1, height1));  
		cvCopy(write, temp1, NULL);  
		cvResetImageROI(temp1);  
		//��ת����map
		// [ m0  m1  m2 ] ===>  [ A11  A12   b1 ]
		// [ m3  m4  m5 ] ===>  [ A21  A22   b2 ]
		float m1[6];  
		int w1 = temp1->width;  
		int h1 = temp1->height;  
		m1[0] = b;  
		m1[1] = a;  
		m1[3] = -m1[1];  
		m1[4] = m1[0];  
		// ����ת��������ͼ���м�  
		m1[2] = w1 * 0.5f;  
		m1[5] = h1 * 0.5f;  
		CvMat M1 = cvMat(2, 3, CV_32F, m1);  
		cvGetQuadrangleSubPix(temp1, Image, &M1);
		cvReleaseImage(&temp1); 

////......................................................................................................
//
//		//Image=rotateImage2(write,degree);//��ת���ͼƬ

//		//		cvNamedWindow("Image", CV_WINDOW_AUTOSIZE);
//		//cvShowImage("Image", Image); //�Ҷ�ͼ
//		//cvWaitKey(0);
//
//
////...........................�ó�������Ŀ��ͼƬ������..........................................................................
//		
		  IplImage *gray2 =  cvCreateImage(cvGetSize(Image), IPL_DEPTH_8U, 1);
		  cvCvtColor(Image, gray2, CV_BGR2GRAY);
		  IplImage *binary2 = cvCreateImage(cvGetSize(gray2), IPL_DEPTH_8U, 1); 
	 
		  cvThreshold(gray2, binary2, 200, 255, CV_THRESH_BINARY); 

  //      cvNamedWindow("binary", CV_WINDOW_AUTOSIZE);
		//cvShowImage("binary", binary2); //binary �Ƕ�ֵ�����ͼƬ
		//cvWaitKey(0);

		IplImage *out = cvCreateImage(cvGetSize(gray2), IPL_DEPTH_8U, 3);
		IplImage *out2 = cvCreateImage(cvGetSize(gray2), IPL_DEPTH_8U, 3);
		CvMemStorage *pcvMStorage2 = cvCreateMemStorage();  
		CvSeq *pcvSeq2 = NULL;  
		cvFindContours(binary2, pcvMStorage2, &pcvSeq2, sizeof(CvContour), CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE, cvPoint(0, 0)); 
		cvDrawContours(out, pcvSeq2, CV_RGB(255,0,0), CV_RGB(0,255,0), 1,2);

		//cvNamedWindow("out1", CV_WINDOW_AUTOSIZE);
		//cvShowImage("out1", out); //�Ҷ�ͼ
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

		double aa=czSize.height*czSize.width;
		double bb=czSize1.height*czSize1.width*3;
		r=aa/bb;
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
			//cvClearMemStorage(c->storage);//�����ڴ�
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


		//cvNamedWindow("out2", CV_WINDOW_AUTOSIZE);
		//cvShowImage("out2", out2); 
		//cvWaitKey(0);
//
//
//
		IplImage *dst_img2 = cvCreateImage(cvGetSize(Image), IPL_DEPTH_8U, 3); 
		
		CvBox2D rect3 = cvMinAreaRect2(pcvSeq2);

		cout<<endl<<endl;
		//cout<<"center3.  "<<rect3.center.x<<"  "<<rect3.center.y<<endl;// ���ӵ�����
		//cout<<"rect3.size.height=="<<rect3.size.width<<"  "<<"rect3.size.width=="<<rect3.size.height<<endl;//���ӵĳ��Ϳ�
		//cout<<"rect3.angle=="<<rect3.angle<<endl;//ע��н� angle ��ˮƽ����ʱ����ת���������ĵ�һ���ߣ������Ǹ߻��ǿ��ļн�


		CvPoint2D32f rect3_pts0[4];
		cvBoxPoints(rect3, rect3_pts0);
		int npt3 = 4;
		CvPoint rect3_pts[4], *ptt = rect3_pts;
	
		for (int rp=0; rp<4; rp++)
		{ 
			rect3_pts[rp]= cvPointFrom32f(rect3_pts0[rp]);
			//cout<<rp<<"  "<<".x"<<rect3_pts[rp].x<<"  "<<".y"<<rect3_pts[rp].y<<endl;//��С��Ӿ��εĶ�������
		}
		cvPolyLine(dst_img2, &ptt, &npt3, 1, 1, CV_RGB(0,255,0), 2);

		//cvNamedWindow("dst_img2", CV_WINDOW_AUTOSIZE);
		//cvShowImage("dst_img2", dst_img2); //dst_img��С��Ӿ��ε�����
		//cvWaitKey(0);	


//..........................�������������.......................................//

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

		//cvNamedWindow("dst4", CV_WINDOW_AUTOSIZE);
		//cvShowImage("dst4", dst4); //Image �Ǽ��ã���ת�����ŵ�ͼƬ
		//cvWaitKey(0);


//............................................................................................
	
		Mat write2,image3;
		
	     IplImage* dst33=cvCreateImage(cvGetSize(Image), src->depth, src->nChannels);
		//IplImage *image11=cvCreateImage(cvGetSize(Image), src->depth, src->nChannels);

		// IplImage *image=cvCreateImage(cvGetSize(Image), src->depth, src->nChannels);

		 cvCopy(Image,dst33,0);

		 write2= Mat(czSize1,CV_8UC3);
		// CvSize cvSize;
		 cvSize4.height=abs(p3.x-p2.x);//x
		 cvSize4.width=abs(p0.y-p2.y);//y

		 cvSetImageROI(dst33,cvRect(p0.x,p0.y,cvSize4.height, cvSize4.width));

		 image3=dst33;
		 Rect rect11(10,10,cvSize4.height, cvSize4.width);
		 image3.copyTo(write2(rect11));
		IplImage * image11=&IplImage(write2);

		// 		cvNamedWindow("dst33", CV_WINDOW_AUTOSIZE);
		//cvShowImage("dst33", dst33); //Image �Ǽ��ã���ת�����ŵ�ͼƬ
		//cvWaitKey(0);


		double angle2 =0 * CV_PI / 180.; 
		double a2 = sin(angle2), b2 = cos(angle2); 
		int width2=image11->width, height2=image11->height;
		//��ת�����ͼ�ߴ�
		int width_rotate2= int(height2 * fabs(a2) + width2 * fabs(b2));  
		int height_rotate2=int(width2 * fabs(a2) + height2 * fabs(b2));   
		//IplImage* image = cvCreateImage(cvSize(width_rotate2, height_rotate2), image11->depth, image11->nChannels); 
		 imagee = cvCreateImage(cvSize(width_rotate2, height_rotate2), image11->depth, image11->nChannels);
		//cvZero(img_rotate);  
		//��֤ԭͼ��������Ƕ���ת����С�ߴ�
		int tempLength2 = sqrt((double)width2 * width2+ (double)height2 *height2) + 10;  
		int tempX2 = (tempLength2 + 1) / 2 - width2 / 2;  
		int tempY2 = (tempLength2 + 1) / 2 - height2 / 2;  
		 //int tempX=0;
		 //int tempY=0;
		IplImage* temp2 = cvCreateImage(cvSize(tempLength2, tempLength2), image11->depth, image11->nChannels);  
		//cvZero(temp);  
		//��ԭͼ���Ƶ���ʱͼ��tmp����
		cvSetImageROI(temp2, cvRect(tempX2, tempY2, width2, height2));  
		cvCopy(image11, temp2, NULL);  
		cvResetImageROI(temp2);  
		//��ת����map
		// [ m0  m1  m2 ] ===>  [ A11  A12   b1 ]
		// [ m3  m4  m5 ] ===>  [ A21  A22   b2 ]
		float m2[6];  
		int w2 = temp2->width;  
		int h2 = temp2->height;  
		m2[0] = b2;  
		m2[1] = a2;  
		m2[3] = -m2[1];  
		m2[4] = m2[0];  
		// ����ת��������ͼ���м�  
		m2[2] = w2 * 0.5f;  
		m2[5] = h2 * 0.5f;  
		CvMat M2 = cvMat(2, 3, CV_32F, m2);  
		cvGetQuadrangleSubPix(temp2, imagee, &M2);
		cvReleaseImage(&temp2); 


		////image=rotateImage2(image11,0);

		//		cvNamedWindow("image", CV_WINDOW_AUTOSIZE);
		//cvShowImage("image", image); //Image �Ǽ��ã���ת�����ŵ�ͼƬ
		//cvWaitKey(0);





		 //cout<<"rect.height==="<<cvSize4.height<<"  "<<"cvSize.width==="<<cvSize4.width<<endl;
		 //cout<<"pic.height==="<<czSize1.height<<"  "<<"pic.width==="<< czSize1.width <<endl;
		 //cout<<"���ƶ�y����==="<<czSize1.height-cvSize4.width-12<<"   "<<"���ƶ�x����"<<czSize1.width-cvSize4.height-12<<endl;

		// //.......................................................................//
		cout<<"�ƶ�ǰ�ľ�������"<<endl;
		cout<<"p0.x"<<" "<<10<<"p0.y"<<" "<<10<<endl;
		cout<<"p1.x"<<p1.x-p0.x+10<<" "<<"p1.y"<<p1.y-p0.y+10<<endl;
		cout<<"p2.x"<<p2.x-p0.x+10<<" "<<"p2.y"<<p2.y-p0.y+10<<endl;
		cout<<"p3.x"<<p3.x-p0.x+10<<" "<<"p3.y"<<p3.y-p0.y+10<<endl;

		{

			//CvPoint  p000,p111,p222,p333;
		p00.x=0+10;
		p00.y=0+10;
		p11.x=p1.x-p0.x+10;
		p11.y=p1.y-p0.y+10;
		p22.x=p2.x-p0.x+10;
		p22.y=p2.y-p0.y+10;
		p33.x=p3.x-p0.x+10;
		p33.y=p3.y-p0.y+10;


		//p000.x=p00.x+dix;
		//p000.y=p00.y+diy;
		//p111.x=p11.x+dix;
		//p111.y=p11.y+diy;
		//p222.x=p22.x+dix;
		//p222.y=p22.y+diy;
		//p333.x=p33.x+dix;
		//p333.y=p33.y+diy;



		//cvLine( image, p00, p11,CV_RGB(0,255,0), 1, 8,0 );
		//cvLine( image, p11, p33,CV_RGB(0,255,0), 1, 8,0 );
		//cvLine( image, p33, p22,CV_RGB(0,255,0), 1, 8,0 );
		//cvLine( image, p22, p00,CV_RGB(0,255,0), 1, 8,0 );

		//cvLine( image, p000, p111,CV_RGB(0,255,0), 1, 8,0 );
		//cvLine( image, p111, p333,CV_RGB(0,255,0), 1, 8,0 );
		//cvLine( image, p333, p222,CV_RGB(0,255,0), 1, 8,0 );
		//cvLine( image, p222, p000,CV_RGB(0,255,0), 1, 8,0 );


		//cout<<endl<<endl;
		//cout<<"ƽ�ƺ�����"<<endl<<endl;
		//cout<<"p0.x==="<<" "<<10+dix<<" "<<"p0.y==="<<10+diy<<endl;
		//cout<<"p1.x==="<<p1.x-p0.x+dix+10<<" "<<"p1.y==="<<p1.y-p0.y+diy+10<<endl;
		//cout<<"p2.x==="<<p2.x-p0.x+dix+10<<" "<<"p2.y==="<<p2.y-p0.y+diy+10<<endl;
		//cout<<"p3.x==="<<p3.x-p0.x+dix+10<<" "<<"p3.y==="<<p3.y-p0.y+diy+10<<endl;



		}
	
		
		//cvNamedWindow("Image", CV_WINDOW_AUTOSIZE);
		//cvShowImage("Image", image); //Image �Ǽ��ã���ת�����ŵ�ͼƬ
		//cvWaitKey(0);	

		 cvReleaseMemStorage(&pcvMStorage); 
		 cvReleaseImage(&dst); 
		 cvReleaseImage(&gray);
		 cvReleaseImage(&binary);
		 cvReleaseImage(&pOutlineImage); 
		 cvReleaseImage(&pOutImage);
		 cvReleaseImage(&dst_img);
		 cvReleaseImage(&zhuan);
		cvReleaseImage(&dst2);
		 cvReleaseImage(&gray1);
         cvReleaseImage(&binary1);
		 cvReleaseImage(&dst_img1);
		 cvReleaseImage(&pSrcImage);
		  cvReleaseMemStorage(&pcvMStorage1); 
		   cvReleaseMemStorage(&pcvMStorage2);
		 cvReleaseImage(&Image);
		 cvReleaseImage(&gray2);
		cvReleaseImage(&binary2);
		 cvReleaseImage(&out);
		 cvReleaseImage(&out2);
		 cvReleaseImage(&dst_img2);
		 cvReleaseImage(&dst4);
		 cvReleaseImage(&dst33);


		//return image;

}
}


void getrectpoints( int dix, int diy, double per, int noise)
{
	if((output_file_ == string("")) || (!file_handler_))
	{
		cout<<endl<<endl;
		cout<<"ƽ�ƺ�����"<<endl<<endl;
		cout<<"dix"<<dix<<endl;
		cout<<"p0.x==="<<" "<<p00.x+dix<<" "<<"p0.y==="<<p00.y+diy<<endl;
		cout<<"p1.x==="<<p11.x+dix<<" "<<"p1.y==="<<p11.y+diy<<endl;
		cout<<"p2.x==="<<p22.x+dix<<" "<<"p2.y==="<<p22.y+diy<<endl;
		cout<<"p3.x==="<<p33.x+dix<<" "<<"p3.y==="<<p33.y+diy<<endl;
		cout<<"ת���Ƕ�==="<<anglee<<endl;
	}
	else
	{
		file_handler_ << main_count_ << " " << p00.x+dix << " " << p00.y+diy << " "
			          << p33.x+dix << " " << p33.y+diy << " "
					  << anglee << " " << per << " " << noise << endl;

		main_count_++;
	}
}



vector<int> getdistance()
{


		 cout<<"rect.height==="<<cvSize4.height<<"  "<<"rect.width==="<<cvSize4.width<<endl;
		 cout<<"pic.height==="<<czSize1.height<<"  "<<"pic.width==="<< czSize1.width <<endl;
		 cout<<"���ƶ�y����==="<<czSize1.height-cvSize4.width-11<<"   "<<"���ƶ�x����"<<czSize1.width-cvSize4.height-11<<endl;
		 vector<int>distance;
		 distance.clear();
		 distance.push_back(czSize1.width-cvSize4.height-11);
		 distance.push_back(czSize1.height-cvSize4.width-11);
		 return distance;
}

#define TWO_PI 6.2831853071795864769252866
double generateGaussianNoise()
{
	static bool hasSpare = false;
	static double rand1, rand2;
 
	if(hasSpare)
	{
		hasSpare = false;
		return sqrt(rand1) * sin(rand2);
	}
 
	hasSpare = true;
 
	rand1 = rand() / ((double) RAND_MAX);
	if(rand1 < 1e-100) rand1 = 1e-100;
	rand1 = -2 * log(rand1);
	rand2 = (rand() / ((double) RAND_MAX)) * TWO_PI;
 
	return sqrt(rand1) * cos(rand2);
}


void AddGaussianNoise(Mat& I,int value)
{
	// accept only char type matrices
	CV_Assert(I.depth() != sizeof(uchar));

	int channels = I.channels();

	int nRows = I.rows;
	int nCols = I.cols * channels;

	if(I.isContinuous()){
		nCols *= nRows;
		nRows = 1;
	}

	int i,j;
	uchar* p;
	for(i = 0; i < nRows; ++i){
		p = I.ptr<uchar>(i);
		for(j = 0; j < nCols; ++j){
			double val = p[j] + generateGaussianNoise() * value;//suggest value in 0-50
			if(val < 0)
				val = 0;
			if(val > 255)
				val = 255;

			p[j] = (uchar)val;

		}
	}

}

Mat salt(cv::Mat& image, int n)//��������
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


void backgroundpicture(const char *save_file,IplImage *src,IplImage *dst1,int Threshold1,int Threshold2 ,double scale,float degree,double per,int dix,int diy,int noise,int value )//Threshold=200
{
	 
	 // IplImage *srcc=originalpicture(src,Threshold1,scale,degree,dix,diy);
	IplImage *srcc=imagee;
	  getrectpoints(dix,diy,per,noise);//���rect������


	  // cvNamedWindow("srcc", CV_WINDOW_AUTOSIZE);
	  //cvShowImage("srcc", srcc); //��ֵͼ
	  //cvWaitKey(0);
	 
	  Mat src1=srcc;
	   
	  Mat dst=dst1;

	  IplImage *gray =  cvCreateImage(cvGetSize(srcc), IPL_DEPTH_8U, 1);
	  cvCvtColor(srcc, gray, CV_BGR2GRAY);
	  IplImage *binary = cvCreateImage(cvGetSize(gray), IPL_DEPTH_8U, 1); 
	 
	
	  cvThreshold(gray, binary, Threshold2, 255, CV_THRESH_BINARY); 
	  //
	  //cvNamedWindow("binary", CV_WINDOW_AUTOSIZE);
	  //cvShowImage("binary", binary); //��ֵͼ
	  //cvWaitKey(0);

	CvMemStorage *pcvMStorage = cvCreateMemStorage();  
    CvSeq *pcvSeq = NULL;  
    cvFindContours(binary, pcvMStorage, &pcvSeq, sizeof(CvContour), CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE, cvPoint(0, 0));    
	IplImage *pOutlineImage = cvCreateImage(cvGetSize(srcc), IPL_DEPTH_8U, 3); 

	 cvDrawContours(pOutlineImage, pcvSeq, CV_RGB(255,0,0), CV_RGB(0,255,0), 1, 2); 

	//cvNamedWindow("pOutlineImage2", CV_WINDOW_AUTOSIZE);
	//cvShowImage("pOutlineImage2", pOutlineImage); //�Ҷ�ͼ
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
		   area=fabs(cvContourArea(pcvSeq,CV_WHOLE_SEQ));
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
			cvClearMemStorage(c->storage);//�����ڴ�
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
	//cvShowImage("pOutlineImage2", pOutlineImage); //�Ҷ�ͼ
	//cvWaitKey(0);

	CvSeq*  tmpseq =pcvSeq;
	for(;tmpseq !=0; tmpseq = tmpseq->h_next)
	{
		double Area=fabs(cvContourArea(tmpseq));
		//cout<<"Area===="<<Area<<endl; 
	}



 //  cvDrawContours(pOutlineImage, pcvSeq, CV_RGB(255,0,0), CV_RGB(0,255,0), 0, 2);  

	//cvNamedWindow("pstrWindowsOutLineTitle", CV_WINDOW_AUTOSIZE);
	//cvShowImage("pstrWindowsOutLineTitle", pOutlineImage); //pOutlineImage����ͼ
	//cvWaitKey(0);

	
	cv::Mat raw_dist( src1.size(), CV_32FC1); 	
	IplImage *raw_dist1=&IplImage(raw_dist);
	Mat mask;mask=Mat::zeros(src1.size(),CV_8UC1);
	IplImage *mask1=&IplImage(mask);



	 //srand((unsigned)time(NULL));
	 IplImage *pSrcImage = cvCreateImage(cvGetSize(srcc), srcc->depth, srcc->nChannels);
	 Mat psrc=pSrcImage;
	 Mat new_image = Mat::zeros( psrc.size(), psrc.type() );
	IplImage *dstImg1=cvCreateImage(cvGetSize(srcc), IPL_DEPTH_8U, 3);
	IplImage *dstImg=&IplImage(dst);
	cvResize(dstImg, dstImg1, CV_INTER_AREA);

	

	//.........................	rgb	.........................
	for( int j = 0; j < src1.rows; j++ )
	{ 
		
     for( int i = 0; i < src1.cols; i++ )
	 { 
		  
         raw_dist.at<float>(j,i) = cvPointPolygonTest( pcvSeq, Point2f(i,j), 0 );
		 
		  if(cvPointPolygonTest( pcvSeq, Point2f(i,j), 0 )>=0)
		  {

			   CvScalar s1=cvGet2D(dstImg1,j+diy,i+dix);//����ͼ
			    CvScalar s3=s1;
			   CvScalar s2=cvGet2D(srcc,j,i);//Ŀ��ͼ
			  // CvScalar color=cvGet2D(srcc,j,i);
			  double a=s1.val[0]*(1-per)+s2.val[0]*per;
			  double b=s1.val[1]*(1-per)+s2.val[1]*per;
			  double c=s1.val[2]*(1-per)+s2.val[2]*per;
			  s3.val[0]=a;
			  s3.val[1]=b;
			  s3.val[2]=c;
			 CvScalar color=s3;
			 //if(noise==1)
			 //{
			 //  if( s3.val[0]<150){color.val[0]=s3.val[0]+(rand()%101/100.0)*50;}
			 //  if(s3.val[0]>150){color.val[0]=s3.val[0]-(rand()%101/100.0)*50;}
			 //  if( s3.val[1]<150){color.val[1]=s3.val[1]+(rand()%101/100.0)*50;}
			 //  if(s3.val[1]>150){color.val[1]=s3.val[1]-(rand()%101/100.0)*50;}
			 //  if( s3.val[2]<150){color.val[2]=s3.val[2]+(rand()%101/100.0)*50;}
			 //  if(s3.val[2]>150){color.val[2]=s3.val[2]-(rand()%101/100.0)*50;}
			 //  //color.val[0]=s2.val[0]*(rand()%101/100.0);
			 //  //color.val[1]=s2.val[1]*(rand()%101/100.0);
			 //  //color.val[2]=s2.val[2]*(rand()%101/100.0);
			 //}
			
			 cvSet2D(pSrcImage,(j+diy)%srcc->height,(i+dix)%srcc->width,color);//�ƶ�Ŀ��ͼƬ		
			 	
			 for( int c = 0; c < 3; c++ )
            {
				if((j+diy)>srcc->height||(i+dix)>srcc->width)
				{cout<<"�ƶ��ߴ糬���߽�"<<endl;}
               // new_image.at<Vec3b>(j+diy,i+dix)[c] = saturate_cast<uchar>( alpha*( psrc.at<Vec3b>(j+diy,i+dix)[c] ) + beta );
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

  
	//cvNamedWindow("pstrWindowsOutLineTitle", CV_WINDOW_AUTOSIZE);
	//cvShowImage("pstrWindowsOutLineTitle", pSrcImage);//ƽ�ƺ��ͼƬ 
	//cvWaitKey(0); 
	//imshow("mask",mask);//��Ĥ	
	//imshow("new_image",new_image);
	// cvWaitKey(0); 


//.........................koutu.........................................
	IplImage *image=&IplImage(new_image);
 
	
	Mat dstt=dstImg1;  
	new_image.copyTo(dstt,mask);//dstt �������տ�ȡ��ͼƬ
	if(noise==1){AddGaussianNoise(dstt,value);}

//.......................................................................
	

	//cvNamedWindow("pstrWindowsOutLineTitle", CV_WINDOW_AUTOSIZE);
	//cvShowImage("pstrWindowsOutLineTitle", dstImg1);//ƽ�ƺ��ͼƬ 
	//cvWaitKey(0);
	  //  imshow("dst",dstt);//����ͼƬ
	        string save_dst;
		save_dst = save_path_ + string(save_file);
		cvSaveImage(save_dst.c_str(),dstImg1);
		cvSaveImage("save.png",dstImg1);//����ͼƬ�Ѿ��������ļ�����
   //     cvWaitKey(0); 


	    cvReleaseMemStorage(&pcvMStorage); 
		 cvReleaseImage(&srcc); 
		 cvReleaseImage(&gray);
		 cvReleaseImage(&binary);
		 cvReleaseImage(&pOutlineImage);
		// cvReleaseImage(&raw_dist1); 
			//cvReleaseImage(&mask1); 
			cvReleaseImage(&pSrcImage); 
			cvReleaseImage(&dstImg1); 
			//cvReleaseImage(&image); 
			//cvReleaseImage(&dstImg); 

}


public:
	 CvSize cvSize4;
     CvSize czSize1;
	 CvPoint p00,p11,p22,p33;//rect������
	 float anglee;//ת���Ƕ�
	float  r;//�������
	IplImage *imagee;

};
