#include <iostream>
#include <cv.h>
#include <highgui.h>

#include "capture.h"
#include <vector>
#include <fstream>

#include "MultiObjectTracker.h"

using namespace std;

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

class ROI
{
  public :
    int   x1;
    int   y1;
    int   x2;
    int   y2;
    float s1;
    float s2;
  ROI( int _x1, int _y1, int _x2, int _y2, float _s1, float _s2)
  {
    x1 = _x1; y1 = _y1; x2 = _x2; y2 = _y2; s1 = _s1; s2 = _s2;
  }
};

/*************************Configurable Parameters*********/

double minScale = 1.0;
double maxScale = 4.0;


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
static bool oneTimeSettingFlag = false;

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

vector<Detection> processFrameFastHog( IplImage* imgIn, vector<ROI> vecROI)
{

  hogImage = convertImageCV2HOG(imgIn);
  //cout << imgIn->width << " " << imgIn->height << endl;

  if( oneTimeSettingFlag == false) oneTimeSettings();

  Timer t;
  t.restart();

  vector<Detection> results;
  for( int i=0; i<vecROI.size(); i++)
  {
    HOGEngine::Instance()->BeginProcess(  hogImage, vecROI[i].x1, vecROI[i].y1,
        vecROI[i].x2, vecROI[i].y2, vecROI[i].s1, vecROI[i].s2);
    HOGEngine::Instance()->EndProcess();

    //printf("Found %d positive results.\n", HOGEngine::Instance()->formattedResultsCount);
    //HOGEngine::Instance()->GetImage(hogImage, HOGEngine::IMAGE_ROI);

    for (int i=0; i<HOGEngine::Instance()->nmsResultsCount; i++)
    {
      if( HOGEngine::Instance()->nmsResults[i].score >=  0)
      {
        /*
        printf("%1.5f %1.5f %4d %4d %4d %4d %4d %4d\n",
            HOGEngine::Instance()->nmsResults[i].score,
            HOGEngine::Instance()->nmsResults[i].scale,
            HOGEngine::Instance()->nmsResults[i].origX,
            HOGEngine::Instance()->nmsResults[i].origY,
            HOGEngine::Instance()->nmsResults[i].x,
            HOGEngine::Instance()->nmsResults[i].y,
            HOGEngine::Instance()->nmsResults[i].width,
            HOGEngine::Instance()->nmsResults[i].height);
        */

        Detection tempDet;
        tempDet.x = HOGEngine::Instance()->nmsResults[i].x;
        tempDet.y = HOGEngine::Instance()->nmsResults[i].y;
        tempDet.width = HOGEngine::Instance()->nmsResults[i].width;
        tempDet.height = HOGEngine::Instance()->nmsResults[i].height;
        tempDet.origX = HOGEngine::Instance()->nmsResults[i].origX;
        tempDet.origY = HOGEngine::Instance()->nmsResults[i].origY;
        tempDet.scale = HOGEngine::Instance()->nmsResults[i].scale;
        tempDet.score = HOGEngine::Instance()->nmsResults[i].score;
//        cvRectangle(imgIn, cvPoint(tempDet.x,tempDet.y),
//            cvPoint(tempDet.x+tempDet.width, tempDet.y+tempDet.height), cvScalar(255,0,0), 2);
        results.push_back( tempDet);
      }
    }
  }

  t.stop(); t.check("Processing time");

  delete hogImage;
  return results;
}

void readScaleMask( vector<ROI>& vecROI, double scaleMask[])
{
  int index;
  double scale;

  std::ifstream file ("scaleMask.txt");

  vector<double> scales;

  if( file)
  {
    for( int i=0; i<1080; i++)
    {
      file >> index >> scale;
      scales.push_back( scale);
      //cout << index << " " << scale << endl;
    }
  }
  file.close();

  for(int i=0; i<1920; i++)
  {
    for( int j=0; j<1080; j++)
      scaleMask[i*1080+j] = scales[i];
  }

  int start = 0;
  while( scales[start] < 1.0) start++;

  int delta = 70;
  int step = 30;
  //int step = 12;
  for( int i=start; i+delta<=1080; i+=step)
  {
    ROI temp = ROI(800, i, 1440, i+delta, scales[i], scales[i+delta]);
    vecROI.push_back(temp);
    //cout << temp.x1 << " " << temp.y1 << " " << temp.x2 << " ";
    //cout << temp.y2 << " " << temp.s1 << " " << temp.s2 << endl;
  }
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

  //cout << "Opening " << argv[1] << endl;
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

    //VideoCapture* temp = (VideoCapture*) capture;
    //cout << "Width : " <<  temp->getWidth() << endl;
    //cout << "Height : " <<  temp->getHeight() << endl;
    //cout << "Duration : " << temp->getDuration() << endl;
    //cout << "FPS : " << temp->getFPS() << endl;

    fps = 25;
  }
  else
  {
    fprintf( stderr, "Couldn't Not Detect Type of Source");
    exit(0);
  }

  if ( !(strncmp( capture->getType(), "VIDEO", 5)))
  {
    VideoCapture* temp = (VideoCapture*) capture;
    temp->play();
    //cout << capture->getStatus();
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

  char key=' ';
  vector<Detection> currDets;
  vector<ROI> vecROI;
  double* scaleMask = new double[capture->getHeight()*capture->getWidth()];

  //vecROI.push_back( ROI(1, 1, image->width, image->height, 1.0, 4.0));
  readScaleMask( vecROI, scaleMask);

  //Construct scale mask and make a multi tracker object

  MultiObjectTracker mulTObj(capture->getWidth(), capture->getHeight(), MD_RGB, scaleMask);

  for( int index = 0; ;index++)
  {
    image->imageData = (char*) capture->getNextFrame();
    if ( image->imageData == NULL) break;

    if( index%15 == 0)
    {
      currDets = processFrameFastHog(image,  vecROI);
      vector<Rect> tempVec;
      int numObjects = currDets.size();
      for( int i=0; i<numObjects; i++)
      {
        int x = currDets[i].x;
        int y = currDets[i].y;
        int w = currDets[i].width;
        int h = currDets[i].height;

        Rect r;
        r.x = x+w/4; r.y = y+h/4;
        r.width = w/2; r.height = h/2;
        tempVec.push_back(r);
      }
      mulTObj.clearObjects();
      mulTObj.addObjects( tempVec);
    }
    else
    {
      mulTObj.processFrame( (UBYTE8*) image->imageData);
      vector<Rect> rects = mulTObj.getObjectsRects();
      for( int i=0; i<rects.size(); i++)
      {
        CvScalar color;
        if( i%3 == 1)
          color = cvScalar(255, 0, 0);
        else if( i%3 == 2)
          color = cvScalar(0, 255, 0);
        else
          color = cvScalar(0, 0, 255);


        cvRectangle(image, cvPoint(rects[i].x, rects[i].y),
                    cvPoint(rects[i].x+rects[i].width,
                    rects[i].y+rects[i].height),
                    color, 2);
      }
    }

    cvShowImage( "Video", image);
    key = cvWaitKey((int) 1000.0 / fps);
    if( key == 27) break;
  }

  HOGEngine::Instance()->FinalizeHOG();
  delete capture;
  return 0;
}
