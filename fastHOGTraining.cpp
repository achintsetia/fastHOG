#include <iostream>
#include <cv.h>
#include <highgui.h>

#include "capture.h"
#include <vector>
#include <fstream>

using namespace std;

#define svmScoreThresh 0.5

class Detection
{
  public :
    float score;
    float scale;
    int origX;
    int origY;
    int x;
    int y;
    int width;
    int height;
};

/*************************FastHog*************************/
#include <stdio.h>

#include <boost/thread/thread.hpp>
#include <fltk/run.h>

#include "HOG/HOGEngine.h"
#include "HOG/HOGImage.h"

#include "Utils/ImageWindow.h"
#include "Utils/Timer.h"

#include "Others/persondetectorwt.tcc"

using namespace HOG;
ImageWindow* fastHOGWindow;
HOGImage* hogImage;
//HOGImage* imageCUDA;
static bool oneTimeSettingFlag = false;

vector<Detection>** vecDetections;

void oneTimeSettings()
{
  HOGEngine::Instance()->InitializeHOG( hogImage->width, hogImage->height,
      "Files//SVM//head_W24x24_C4x4_N2x2_G4x4_HeadSize16x16.alt");
  oneTimeSettingFlag = true;
}

HOGImage* convertImageCV2HOG( IplImage* imgIn)
{
  int w = imgIn->width;
  int h = imgIn->height;
  HOGImage* tempIm = new HOGImage( w, h);
  int wStep = imgIn->widthStep;

  unsigned char* destPixels = tempIm->pixels;
  unsigned char* imageData = (unsigned char*) imgIn->imageData;

  for( int i=0; i<h; i++)
  {
    for(int j=0; j<w; j++)
    {
      *(destPixels + i*w*4 + 4*j + 0) = *(imageData + i*wStep + 3*j + 2);
      *(destPixels + i*w*4 + 4*j + 1) = *(imageData + i*wStep + 3*j + 1);
      *(destPixels + i*w*4 + 4*j + 2) = *(imageData + i*wStep + 3*j + 0);
      *(destPixels + i*w*4 + 4*j + 3) = 0;
    }
  }
  return tempIm;
}

void processFrameFastHog( IplImage* imgIn)
{

  hogImage = convertImageCV2HOG(imgIn);
  cout << imgIn->width << " " << imgIn->height << endl;


  if( oneTimeSettingFlag == false) oneTimeSettings();

  Timer t;
  t.restart();
  HOGEngine::Instance()->BeginProcess(hogImage);
  HOGEngine::Instance()->EndProcess();
  t.stop(); t.check("Processing time");

  printf("Found %d positive results.\n", HOGEngine::Instance()->formattedResultsCount);

  for (int i=0; i<HOGEngine::Instance()->nmsResultsCount; i++)
  {
    if( HOGEngine::Instance()->nmsResults[i].score >=  svmScoreThresh)
    {
      printf("%1.5f %1.5f %4d %4d %4d %4d %4d %4d\n",
          HOGEngine::Instance()->nmsResults[i].score,
          HOGEngine::Instance()->nmsResults[i].scale,
          HOGEngine::Instance()->nmsResults[i].origX,
          HOGEngine::Instance()->nmsResults[i].origY,
          HOGEngine::Instance()->nmsResults[i].x,
          HOGEngine::Instance()->nmsResults[i].y,
          HOGEngine::Instance()->nmsResults[i].width,
          HOGEngine::Instance()->nmsResults[i].height);

      int x = HOGEngine::Instance()->nmsResults[i].x;
      int y = HOGEngine::Instance()->nmsResults[i].y;
      int w = HOGEngine::Instance()->nmsResults[i].width;
      int h = HOGEngine::Instance()->nmsResults[i].height;
      cvRectangle(imgIn, cvPoint(x,y), cvPoint(x+w, y+h), cvScalar(255,0,0), 2);

      Detection det;
      det.score = HOGEngine::Instance()->nmsResults[i].score;
      det.scale = HOGEngine::Instance()->nmsResults[i].scale;
      det.origX = HOGEngine::Instance()->nmsResults[i].origX;
      det.origY = HOGEngine::Instance()->nmsResults[i].origY;
      det.x = HOGEngine::Instance()->nmsResults[i].x;
      det.y = HOGEngine::Instance()->nmsResults[i].y;
      det.width = HOGEngine::Instance()->nmsResults[i].width;
      det.height = HOGEngine::Instance()->nmsResults[i].height;

      if( x >= 0 && x < imgIn->width && y>=0 && y < imgIn->height)
        vecDetections[y][x].push_back(det);
    }
  }

  printf("Drawn %d positive results.\n", HOGEngine::Instance()->nmsResultsCount);

  delete hogImage;
}


/***************************CaptureLibFunctions*************/
Capture* capture = NULL;

int main( int argc, char** argv)
{
  if ( argc != 2 )
  {
    cout << "Usage :" << argv[0] << " <video/image filename or camera device>" << endl;
    exit(0);
  }

  cout << "Opening " << argv[1] << endl;
  capture = new Capture( argv[1]);

  char type[32];
  strcpy( type, capture->getType());
  cout << "Type :" << type << endl;

  float fps = 100;

  cvNamedWindow( "Video", 1);

  if ( !strncmp( type, "VIDEO", 5))
  {
    delete capture;
    capture = new VideoCapture( argv[1]);

    VideoCapture* temp = (VideoCapture*) capture;

    //cout << "Width : " <<  temp->getWidth() << endl;
    //cout << "Height : " <<  temp->getHeight() << endl;
    //cout << "Duration : " << temp->getDuration() << endl;
    //cout << "FPS : " << temp->getFPS() << endl;

    fps = 25;
  }
  else
  {
    fprintf( stderr, "Couldn't Not Detect Type of Source");
    fprintf( stderr, " or No Handeller Available\n");
    exit(0);
  }

  if ( !(strncmp( capture->getType(), "VIDEO", 5)))
  {
    VideoCapture* temp = (VideoCapture*) capture;
    temp->play();
    cout << capture->getStatus();
  }

  IplImage* image = NULL;

  if (( capture->getWidth() > 0) && (capture->getHeight() > 0))
  {
    image = cvCreateImageHeader( cvSize(capture->getWidth(),
          capture->getHeight()), IPL_DEPTH_8U, 3);
  }
  else
  {
    cout << capture->getStatus();
    return 0;
  }
  //Assuming height 1100 and width 2000
  vecDetections =  new vector<Detection>* [1100];
  for( int i=0; i<1100; i++)
    vecDetections[i] = new vector<Detection>[2000];


  char key=' ';
  for( int i = 0; ;)
  {
    image->imageData = (char*) capture->getNextFrame();
    processFrameFastHog(image);

    if ( image->imageData == NULL) break;

    cvShowImage( "Video", image);

    key = cvWaitKey((int) 1000.0 / fps);
    if (( key == '\n') && ( i == 0))
    {
      i = 1;
      if ( !(strncmp( capture->getType(), "VIDEO", 5)))
      {
        VideoCapture* temp = (VideoCapture*) capture;
        temp->pause();
      }
    }
    else if( key == 27) break;
  }

  fstream outfile;
  outfile.open("dump.txt", ios_base::out);

  for(int i=0; i<capture->getHeight(); i++)
    for(int j=0; j<capture->getWidth(); j++)
    {
      vector<Detection> detVec = vecDetections[i][j];
      for(int k=0; k<detVec.size(); k++)
      {
        outfile << detVec[k].y << " " << detVec[k].x << " " << detVec[k].width;
        outfile << " " << detVec[k].height << " " << detVec[k].scale << " " << detVec[k].score << endl;
      }
    }

  outfile.close();

  HOGEngine::Instance()->FinalizeHOG();
  delete capture;
  return 0;
}
