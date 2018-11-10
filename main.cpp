        /*
                           _ooOoo_
                          o8888888o
                          88" . "88
                          (| -_- |)
                          O\  =  /O
                       ____/`---'\____
                     .'  \\|     |//  `.
                    /  \\|||  :  |||//  \
                   /  _||||| -:- |||||-  \
                   |   | \\\  -  /// |   |
                   | \_|  ''\---/''  |   |
                   \  .-\__  `-`  ___/-. /
                 ___`. .'  /--.--\  `. . __
              ."" '<  `.___\_<|>_/___.'  >'"".
             | | :  `- \`.;`\ _ /`;.`/ - ` : | |
             \  \ `-.   \_ __\ /__ _/   .-` /  /
        ======`-.____`-.___\_____/___.-`____.-'======
                           `=---='
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        */

#include <iostream>
#include <string>
#include <vector>
#include <utility>
#include <typeinfo>
#include "math.h"
#include "limits.h"
#include <opencv2/features2d.hpp>
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <opencv2/opencv.hpp>
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/core/core.hpp"
#include <opencv2/features2d/features2d.hpp>
#include <tiny_dnn/tiny_dnn.h>

using namespace cv;
using namespace std;

#define PI 3.14159265
double match_threshold = 0.2;
double alpha_threshold = 1;
int rm_cone_threshold = 0;
int resultSize = 752;
vector<Scalar> COLOR = {{255,255,0},{0,255,255},{0,165,255},{0,0,255}};

int global_i;
bool skip_frame = 0;

struct KP
{
	Point3d pt;
	int id;
};

struct Cone{
	int m_x, m_y;
	double m_prob;
	int m_label;
};

int m_width = 672;
int m_height = 376;
float m_maxZ = 8.2;
int m_patchSize = 64;
float m_threshold = 0.5;
tiny_dnn::network<tiny_dnn::sequential> m_model;

void show_image(cv::Mat img){
	namedWindow("img", WINDOW_NORMAL);
	imshow("img", img);
	waitKey(0);
}

// void stereoSGBM(cv::Mat& disp, cv::Mat imgL, cv::Mat imgR){
//   cv::Mat grayL, grayR, dispL, dispR;
//
//   cv::cvtColor(imgL, grayL, 6);
//   cv::cvtColor(imgR, grayR, 6);
//
//   cv::Ptr<cv::StereoSGBM> sgbm = cv::StereoSGBM::create();
//   int blockSize=9;
// 	int cn = 5;
// 	// sgbm->setPreFilterCap(63);
//   sgbm->setBlockSize(blockSize);
//   sgbm->setP1(4*cn*blockSize*blockSize);
//   sgbm->setP2(32*cn*blockSize*blockSize);
//   // sgbm->setMinDisparity(0);
//   sgbm->setNumDisparities(32);
//   sgbm->setUniquenessRatio(10);
//   // sgbm->setSpeckleWindowSize(100);
//   // sgbm->setSpeckleRange(32);
//   // sgbm->setDisp12MaxDiff(1);
//   sgbm->compute(grayL, grayR, dispL);
//
//   disp = dispL/16;
//
// 	cv::Mat disp8;
// 	disp.convertTo(disp8, CV_8U, 255/32);
//
// 	// show_image(disp8);
// 	// cv::imwrite("tmp/stereoSGBM.png", disp8);
// }

void stereoBM(cv::Mat& disp, cv::Mat imgL, cv::Mat imgR){
  cv::Mat grayL, grayR, dispL, dispR;

  cv::cvtColor(imgL, grayL, 6);
  cv::cvtColor(imgR, grayR, 6);

  cv::Ptr<cv::StereoBM> sbmL = cv::StereoBM::create();
  sbmL->setBlockSize(19);
  sbmL->setNumDisparities(32);
  sbmL->setUniquenessRatio(15);
  sbmL->compute(grayL, grayR, dispL);

  disp = dispL/16;

	// cv::Mat disp8;
	// disp.convertTo(disp8, CV_8U, 255/32);
	// show_image(disp8)
	// cv::imwrite("tmp/stereoBM.png", disp8);
}

void reconstruction(cv::Mat img, cv::Mat& Q, cv::Mat& rectified, cv::Mat& XYZ){
  //camera matrix of the left image sensor
  cv::Mat mtxLeft = (cv::Mat_<double>(3, 3) <<
    349.891, 0, 334.352,
    0, 349.891, 187.937,
    0, 0, 1);
  //distortion parameters of the left lens
  cv::Mat distLeft = (cv::Mat_<double>(5, 1) << -0.173042, 0.0258831, 0, 0, 0);
  cv::Mat mtxRight = (cv::Mat_<double>(3, 3) <<
    350.112, 0, 345.88,
    0, 350.112, 189.891,
    0, 0, 1);
  cv::Mat distRight = (cv::Mat_<double>(5, 1) << -0.174209, 0.026726, 0, 0, 0);
  //rotation between two image sensor
  cv::Mat rodrigues = (cv::Mat_<double>(3, 1) << -0.0132397, 0.021005, -0.00121284);

  cv::Mat R;
  //convert to rotation matrix
  cv::Rodrigues(rodrigues, R);
  cv::Mat T = (cv::Mat_<double>(3, 1) << -0.12, 0, 0);
  cv::Size stdSize = cv::Size(m_width, m_height);

  int width = img.cols;
  int height = img.rows;
  //split stereo pair into left and right images
  cv::Mat imgL(img, cv::Rect(0, 0, width/2, height));
  cv::Mat imgR(img, cv::Rect(width/2, 0, width/2, height));

  //resize image to save computation load. Capturing higher resolution image and resize it has better quality than capturing low resolution image
  cv::resize(imgL, imgL, stdSize);
  cv::resize(imgR, imgR, stdSize);

  //rectification
  cv::Mat R1, R2, P1, P2;
  cv::Rect validRoI[2];
  cv::stereoRectify(mtxLeft, distLeft, mtxRight, distRight, stdSize, R, T, R1, R2, P1, P2, Q,
    cv::CALIB_ZERO_DISPARITY, 0.0, stdSize,& validRoI[0],& validRoI[1]);
  cv::Mat rmap[2][2];
  cv::initUndistortRectifyMap(mtxLeft, distLeft, R1, P1, stdSize, CV_16SC2, rmap[0][0], rmap[0][1]);
  cv::initUndistortRectifyMap(mtxRight, distRight, R2, P2, stdSize, CV_16SC2, rmap[1][0], rmap[1][1]);
  cv::remap(imgL, imgL, rmap[0][0], rmap[0][1], cv::INTER_LINEAR);
  cv::remap(imgR, imgR, rmap[1][0], rmap[1][1], cv::INTER_LINEAR);

  cv::Mat disp;
  //compute disparity map by stereoBM
	stereoBM(disp, imgL, imgR);

  imgL.copyTo(rectified);

  //convert to 3d position
  cv::reprojectImageTo3D(disp, XYZ, Q);
}

void convertImage(cv::Mat img, int w, int h, tiny_dnn::vec_t& data){
  cv::Mat resized;
  cv::resize(img, resized, cv::Size(w, h));
  data.resize(w * h * 3);
  //convert cv mat format to tiny-dnn format
  for (int c = 0; c < 3; ++c) {
    for (int y = 0; y < h; ++y) {
      for (int x = 0; x < w; ++x) {
       data[c * w * h + y * w + x] =
         float(resized.at<cv::Vec3b>(y, x)[c] / 255.0);
      }
    }
  }
}

void CNN(const std::string& dictionary, tiny_dnn::network<tiny_dnn::sequential>& model) {
  using conv    = tiny_dnn::convolutional_layer;
  using fc      = tiny_dnn::fully_connected_layer;
  using tanh    = tiny_dnn::tanh_layer;
  using relu    = tiny_dnn::relu_layer;
  using softmax = tiny_dnn::softmax_layer;

  tiny_dnn::core::backend_t backend_type = tiny_dnn::core::default_engine();

  //network architecture
  model << conv(64, 64, 4, 3, 16, tiny_dnn::padding::valid, true, 2, 2, backend_type) << tanh()
     << conv(31, 31, 3, 16, 16, tiny_dnn::padding::valid, true, 2, 2, backend_type) << tanh()
     << conv(15, 15, 3, 16, 32, tiny_dnn::padding::valid, true, 2, 2, backend_type) << tanh()
     << conv(7, 7, 3, 32, 32, tiny_dnn::padding::valid, true, 2, 2, backend_type) << tanh()
     << fc(3 * 3 * 32, 128, true, backend_type) << relu()
     << fc(128, 4, true, backend_type) << softmax(4);

  std::ifstream ifs(dictionary.c_str());
  if (!ifs.good()){
    std::cout << "CNN model does not exist!" << std::endl;
    return;
  }
  ifs >> model;
}

void imRegionalMax(std::vector<Cone>& cones, size_t label, cv::Mat input, int nLocMax, double threshold, int minDistBtwLocMax)
{
  //find the local maxima within a certain size of region
    cv::Mat scratch = input.clone();
    for (int i = 0; i < nLocMax; i++) {
        cv::Point location;
        double maxVal;
        cv::minMaxLoc(scratch, NULL, &maxVal, NULL, &location);
        if (maxVal > threshold) {
            int col = location.x;
            int row = location.y;
            Cone cone;
            cone.m_x = col;
            cone.m_y = row;
            cone.m_prob = maxVal;
            cone.m_label = label;
            cones.push_back(cone);
            int r0 = (row-minDistBtwLocMax > -1 ? row-minDistBtwLocMax : 0);
            int r1 = (row+minDistBtwLocMax < scratch.rows ? row+minDistBtwLocMax : scratch.rows-1);
            int c0 = (col-minDistBtwLocMax > -1 ? col-minDistBtwLocMax : 0);
            int c1 = (col+minDistBtwLocMax < scratch.cols ? col+minDistBtwLocMax : scratch.cols-1);
            for (int r = r0; r <= r1; r++) {
                for (int c = c0; c <= c1; c++) {
                    if (sqrt((r-row)*(r-row)+(c-col)*(c-col)) <= minDistBtwLocMax) {
                      scratch.at<double>(r,c) = 0.0;
                    }
                }
            }
        } else {
            break;
        }
    }
}


cv::Point3f median(std::vector<cv::Point3f> vec3) {
  //compute the median 3d point from a group of 3d points
  size_t size = vec3.size();
  float tvecan[3];
  std::vector<float> vec[3];
  for (size_t i = 0; i < size; i++){
    vec[0].push_back(vec3[i].x);
    vec[1].push_back(vec3[i].y);
    vec[2].push_back(vec3[i].z);
  }

  for (size_t i = 0; i < 3; i++){
    std::sort(vec[i].begin(), vec[i].end());
    if (size % 2 == 0) {
      tvecan[i] = (vec[i][size/2-1] + vec[i][size/2])/2;
    }
    else
      tvecan[i] = vec[i][size/2];
  }

  return cv::Point3f(tvecan[0],0.9f,tvecan[2]);
}

void filterKeypoints(std::vector<cv::Point3f>& point3Ds){
  std::vector<KP> data;
  double radius = 0.5;
  unsigned int max_neighbours = 100;

  for(size_t i = 0; i < point3Ds.size(); i++){
    data.push_back(KP{point3Ds[i],-1});
  }

  cv::Mat source = cv::Mat(point3Ds).reshape(1);
  point3Ds.clear();
  cv::Point3f point3D;
  int groupId = 0;

  for(size_t j = 0; j < data.size()-1; j++)
  {
    if(data[j].id == -1){
      std::vector<float> vecQuery(3);
      vecQuery[0] = data[j].pt.x;
      vecQuery[1] = data[j].pt.y;
      vecQuery[2] = data[j].pt.z;
      std::vector<int> vecIndex;
      std::vector<float> vecDist;

      cv::flann::KDTreeIndexParams indexParams(2);
      cv::flann::Index kdtree(source, indexParams);
      cv::flann::SearchParams params(1024);
      kdtree.radiusSearch(vecQuery, vecIndex, vecDist, radius, max_neighbours, params);

      int num = 0;
      for(size_t i = 0; i < vecIndex.size(); i++){
        if(vecIndex[i]!=0)
          num++;
      }
      for (size_t i = 1; i < vecIndex.size(); i++){
        if (vecIndex[i] == 0 && vecIndex[i+1] != 0){
          num++;
        }
      }
      if (num == 0){
        if (data[j].id == -1){
          data[j].id = groupId++;
          point3D = data[j].pt;
        }
      }
      else{
        std::vector<KP> groupAll;
        std::vector<int> filteredIndex;
        std::vector<cv::Point3f> centerPoints;
        for (int v = 0; v < num; v++){
          groupAll.push_back(data[vecIndex[v]]);
          filteredIndex.push_back(vecIndex[v]);
        }

        int noGroup = 0;
        for(size_t i = 0; i < groupAll.size(); i++){
          if(groupAll[i].id == -1)
            noGroup++;
        }

        if (noGroup > 0){
          for (size_t k = 0; k < filteredIndex.size(); k++)
          {
            if (data[filteredIndex[k]].id == -1)
            {
              data[filteredIndex[k]].id = groupId;
              centerPoints.push_back(data[vecIndex[k]].pt);
            }
          }
          groupId++;
          point3D = median(centerPoints);
        }
        else{
          data[j].id = data[vecIndex[0]].id;
          point3D = data[j].pt;
        }
      }
			// cout << point3D.z << endl;
      if(std::isnan(point3D.x) || std::isnan(point3D.y) || std::isnan(point3D.z))
        continue;
      point3Ds.push_back(point3D);
    }
  }
}

int xyz2xy(cv::Mat Q, cv::Point3f xyz, cv::Point& xy, float radius){
  float X = xyz.x;
  float Y = xyz.y;
  float Z = xyz.z;
  float Cx = float(-Q.at<double>(0,3));
  float Cy = float(-Q.at<double>(1,3));
  float f = float(Q.at<double>(2,3));
  float a = float(Q.at<double>(3,2));
  float b = float(Q.at<double>(3,3));
  float d = (f - Z * b ) / ( Z * a);
  xy.x = int(X * ( d * a + b ) + Cx);
  xy.y = int(Y * ( d * a + b ) + Cy);
  return int(radius * ( d * a + b ));
}

cv::Mat forwardDetectionORB(int i){
	String img_path = "images/"+to_string(i)+".png";
	String csv_path = "tmp/results/"+to_string(i)+".csv";
	std::ofstream file;
	file.open(csv_path.c_str());

	cv::Mat img = cv::imread(img_path);

	std::vector<Cone> cones;
  std::vector<tiny_dnn::tensor_t> inputs;
  std::vector<int> verifiedIndex;
  std::vector<cv::Point> candidates;
  std::vector<cv::Scalar> colors;
  colors.push_back(cv::Scalar(0,0,0));
  colors.push_back(cv::Scalar(255,0,0));
  colors.push_back(cv::Scalar(0,255,255));
  colors.push_back(cv::Scalar(0,165,255));
  std::string labels[] = {"background", "blue", "yellow", "orange", "orange2"};
  int resultWidth = m_height;
  int resultHeight = m_height;
  cv::Mat result = cv::Mat::zeros(resultWidth,resultHeight,CV_8UC3);
  double resultResize = 15;

  cv::Mat Q, XYZ, imgRoI, imgSource;
  reconstruction(img, Q, img, XYZ);

  img.copyTo(imgSource);
  int rowT = 190;
  int rowB = 320;
  imgRoI = imgSource.rowRange(rowT, rowB);

  cv::Ptr<cv::ORB> detector = cv::ORB::create(100);
  std::vector<cv::KeyPoint> keypoints;
  detector->detect(imgRoI, keypoints);
  if(keypoints.size()==0)
    return img;

  cv::Mat probMap[4] = cv::Mat::zeros(m_height, m_width, CV_64F);

  std::vector<cv::Point3f> point3Ds;
  cv::Point point2D;
  std::vector<cv::Point> positions;
  for(size_t i = 0; i < keypoints.size(); i++){
    cv::Point position(int(keypoints[i].pt.x), int(keypoints[i].pt.y)+rowT);
    cv::Point3f point3D = XYZ.at<cv::Point3f>(position);
    // if(point3D.y>0.7 && point3D.y<0.85 && point3D.z > 0 && point3D.z < m_maxZ){
      point3Ds.push_back(point3D);
      positions.push_back(position);
    // }
  }
  if(point3Ds.size()==0)
    return img;
  filterKeypoints(point3Ds);
  for(size_t i = 0; i < point3Ds.size(); i++){
    int radius = xyz2xy(Q, point3Ds[i], point2D, 0.3f);
    int x = point2D.x;
    int y = point2D.y;

    cv::Rect roi;
    roi.x = std::max(x - radius, 0);
    roi.y = std::max(y - radius, 0);
    roi.width = std::min(x + radius, img.cols) - roi.x;
    roi.height = std::min(y + radius, img.rows) - roi.y;

    if(0 < roi.width && 0 < roi.height && radius > 0){
      cv::Mat patchImg = imgSource(roi);
      tiny_dnn::vec_t data;
      convertImage(patchImg, m_patchSize, m_patchSize, data);
      inputs.push_back({data});
      verifiedIndex.push_back(i);
      candidates.push_back(cv::Point(x,y));
    }
  }

  if(inputs.size()>0){
		// time_t currentTime = clock();
			auto prob = m_model.predict(inputs);
		// 	double dur = (double)(clock()-currentTime);
		// cout << dur/1000 << endl;

    for(size_t i = 0; i < inputs.size(); i++){
      size_t maxIndex = 0;
      double maxProb = prob[i][0][0];
      for(size_t j = 1; j < 4; j++){
        if(prob[i][0][j] > maxProb){
          maxIndex = j;
          maxProb = prob[i][0][j];
        }
      }
      int x = candidates[i].x;
      int y = candidates[i].y;
      if(maxIndex > 0)
        probMap[maxIndex].at<double>(y,x) = maxProb;
    }
    for(size_t i = 0; i < 4; i++){
      imRegionalMax(cones, i, probMap[i], 10, m_threshold, 10);
    }

    for(size_t i = 0; i < cones.size(); i++){
      int x = cones[i].m_x;
      int y = cones[i].m_y;
      double maxProb = cones[i].m_prob;
      int maxIndex = cones[i].m_label;
      cv::Point position(x, y);
      cv::Point3f point3D = XYZ.at<cv::Point3f>(position);
      std::string labelName = labels[maxIndex];
      cv::Point position_tmp;
      int radius = xyz2xy(Q, point3D, position_tmp, 0.3f);

      if(radius<=0){
        continue;
      }

      int xt = int(point3D.x * float(resultResize) + resultWidth/2);
      int yt = int(point3D.z * float(resultResize));
      cv::circle(img, position, radius, colors[maxIndex], 2);
      if (xt >= 0 && xt <= resultWidth && yt >= 0 && yt <= resultHeight){
        cv::circle(result, cv::Point (xt,yt), 6, colors[maxIndex], -1);
      }
			file << x << "," << y << "," << labels[maxIndex] << "," << point3D.x << "," << point3D.y << "," << point3D.z << std::endl;
    }
  }

  for(size_t i = 0; i < positions.size(); i++){
    cv::circle(img, positions[i], 2, cv::Scalar (255,255,255), -1);
  }

  cv::line(img, cv::Point(0,rowT), cv::Point(m_width,rowT), cv::Scalar(0,0,255), 2);
  cv::line(img, cv::Point(0,rowB), cv::Point(m_width,rowB), cv::Scalar(0,0,255), 2);

  // cv::Mat outImg;
  // cv::flip(result, result, 0);
  // cv::hconcat(img,result,outImg);

	// show_image(img);
	// cv::imwrite(result_path, img);
	file.close();
	return img;
}




double computeResidual(Point3d pt1, Point3d pt2){
	return pow((pow ((pt1.x-pt2.x),2) + pow ((pt1.y-pt2.y),2)),0.5);
}


void matchFeatures(vector<KP> featureLast, vector<KP> featureNext, vector<DMatch> &matched){
	matched.clear();
	double res, minRes;
	int featureLastRow = featureLast.size();
	int featureNextRow = featureNext.size();
	int index;
	int matchSize = 1000;
	double matchResize = 300;
	Mat match = Mat::zeros(matchSize, matchSize, CV_8UC3);

	for(int i = 0; i < featureNextRow; i++){
		minRes = match_threshold;
		for(int j = 0; j < featureLastRow; j++){
		    if(featureNext[i].id == featureLast[j].id){
		        res = computeResidual(featureNext[i].pt, featureLast[j].pt);//check residuals, find the smallest one, save it
				if(res < minRes){
				    minRes = res;
				    index = j;
		    	}
			}
		}
		if(minRes < match_threshold){
			matched.push_back(DMatch(index,i,minRes));
			int x = int(featureLast[index].pt.x * matchResize + matchSize/2);
			int y = int(featureLast[index].pt.y * matchResize + matchSize/4);
			int x1 = int(featureNext[i].pt.x * matchResize + matchSize/2);
			int y1 = int(featureNext[i].pt.y * matchResize + matchSize/4);
			circle(match, Point (x,y), 5, COLOR[featureNext[i].id], -1);
			circle(match, Point (x1,y1), 3, COLOR[featureNext[i].id], -1);
			line(match, Point(x, y), Point(x1, y1), Scalar(255, 255, 255), 1);

			// cout << global_i << ": " << index << " " << i << " " << minRes << endl;
		}
	}

	// flip(match, match, 0);
 //    namedWindow("match", WINDOW_NORMAL);
	// imshow("match", match);
	// waitKey(0);
}

//void drawmatches(vector<KP> featureLast, vector<KP> featureNext, vector<DMatch> &matched, )

// void matchFeaturesAffine(Mat affine, vector<KP> featureLast,
//                   vector<KP> featureNext, vector<DMatch> &matched){
// 	double match_threshold = 0.1;
// 	double res, minRes;
// 	int featureLastRow = featureLast.size();
// 	int featureNextRow = featureNext.size();
// 	int index;
// 	matched.clear();

// 	for(int i = 0; i < featureNextRow; i++){
// 		minRes = match_threshold;
// 		for(int j = 0; j < featureLastRow; j++){
// 		    if(featureNext[i].id == featureLast[j].id){
// 		  	    Point3d affine_pt(Mat(affine*Mat(featureLast[j].pt)));
// 		        res = computeResidual(featureNext[i].pt, affine_pt);//check residuals, find the smallest one, save it
// 		        if(res < minRes){
// 		            minRes = res;
// 		            index = j;
// 		        }
// 		    }
// 		}
// 		if(minRes < match_threshold){
// 		    matched.push_back(DMatch(index,i,minRes));
// 		    // cout << index << " " << i << " " << imageId << " " << minRes << endl;
// 		}
// 	}
// }

void get_matched_points(//根据matched,返回匹配时两张图分别的坐标
	vector<KP>& p1,
	vector<KP>& p2,
	vector<Vec3b>& c1,
	vector<Vec3b>& c2,
	vector<DMatch>& matched,
	vector<Point3d>& out_p1,
	vector<Point3d>& out_p2,
	vector<Vec3b>& out_c1,
	vector<Vec3b>& out_c2
	)
{
	out_p1.clear();
	out_p2.clear();
	out_c1.clear();
	out_c2.clear();
	for (int i = 0; i < matched.size(); ++i)
	{
		out_p1.push_back(p1[matched[i].queryIdx].pt);
		out_p2.push_back(p2[matched[i].trainIdx].pt);
		out_c1.push_back(c1[matched[i].queryIdx]);
		out_c2.push_back(c2[matched[i].trainIdx]);
	}
}

void reprojectionErrors(Mat affine_tmp, vector<Point3d> p1, vector<Point3d> p2, double& error){
	for(int i=0; i<p1.size();i++){
		Mat p1est = affine_tmp.inv()*Mat(p2[i]);
		error += pow(pow(p1est.at<double>(0,0)-p1[i].x,2)+pow(p1est.at<double>(1,0)-p1[i].y,2),0.5f);
	}
}

void estimateTransform2D(vector<Point3d> p1, vector<Point3d> p2, Mat& best_affine, double& min_error){
	if(p1.size()<2){
		cout << global_i << ": too few points to estimate transformation!" << endl;
		return;
	}
	best_affine.release();
	min_error = 100;

	for(int i = 0; i < p1.size()-1; i++){
		for(int k = i+1; k<p1.size();k++){
			vector<double> alpha;
			//vector<Mat>affine_forransac-temp;
			double a = pow(p1[k].x-p1[i].x,2)+pow(p1[i].y-p1[k].y,2);
			double b = 2*(p2[i].x-p2[k].x)*(p1[i].y-p1[k].y);
			double c = pow(p2[k].x-p2[i].x,2)-pow(p1[k].x-p1[i].x,2);
			alpha.push_back(asin((-b+pow(pow(b,2)-4*a*c,0.5f))/(2*a)));
			alpha.push_back(asin((-b-pow(pow(b,2)-4*a*c,0.5f))/(2*a)));
			for(int j = 0; j < 2; j++){
				if(alpha[j]>alpha_threshold||alpha[j]<-alpha_threshold||isnan(alpha[j])){continue;}
				double cosa = cos(alpha[j]);
				double sina = sin(alpha[j]);
				double error = 0;
				double tx = p2[i].x-cosa*p1[i].x+sina*p1[i].y;
				double ty = p2[i].y-sina*p1[i].x-cosa*p1[i].y;
				Mat affine_tmp(Matx33d(cosa,-sina,tx,sina,cosa,ty,0,0,1));

				//cout<<"starts p"<<i<<"loop"<<j<<endl;
				reprojectionErrors(affine_tmp, p1, p2, error);
				// cout<<"done p"<<i<<"loop"<<j<<endl;
				// int inlierscount_if(error.begin(),error.end(),m_compare);
				// cout<<"inliers size"<<inliers<<endl;
				// if (inliers>most_inliers){
				// 	most_inliers=inliers;
				// 	best_affine=affine_tmp;
				// }
				if (error < min_error){
					min_error = error;
					best_affine = affine_tmp;
				}
			}
		}
	}
}


void reconstruct(
	vector<KP>& last_keypoints,
	vector<KP>& next_keypoints,
	vector<Vec3b>& last_colors,
	vector<Vec3b>& next_colors,
	vector<DMatch>& matched,
	vector<Vec3b>& c1,
	vector<Point3d>& p2,
	Mat& affine,
	double& min_error)
{
	matchFeatures(last_keypoints, next_keypoints, matched);
	vector<Point3d> p1;
	vector<Vec3b> c2;
	get_matched_points(last_keypoints, next_keypoints, last_colors, next_colors, matched, p1, p2, c1, c2);
	estimateTransform2D(p1, p2, affine, min_error);

	if (affine.rows == 0){
    	cout << "Fail to estimate affine transformation, number of points: " << p1.size() << endl;
    }
}

void init_structure(
	vector<vector<KP>>& keypoints_for_all,
	vector<vector<Vec3b>>& colors_for_all,
	vector<Point3d>& structure,
	vector<Vec3b>& colors,
	vector<vector<int>>& correspond_struct_idx,
	vector<Mat>& affines)
{
	vector<DMatch> matched, matched_tmp;
	Mat affine, affine_tmp;
	vector<Vec3b> colors_tmp;
	vector<Point3d> p2, p2_tmp;
	double min_error1, min_error2;

    reconstruct(keypoints_for_all[0], keypoints_for_all[1], colors_for_all[0], colors_for_all[1], matched, colors, p2, affine, min_error1);


    affine = affine*affines.back();
	affines.push_back(affine);

	for(int i = 0; i < p2.size(); i++){
		structure.push_back(Point3d(Mat(affine.inv()*Mat(p2[i]))));
	}

	int idx = 0;
	for (int i = 0; i < matched.size(); ++i)
	{
		correspond_struct_idx[0][matched[i].queryIdx] = idx;
		correspond_struct_idx[1][matched[i].trainIdx] = idx;
		++idx;
	}
}

void fusion_structure(
	vector<DMatch>& matched,
	vector<int>& struct_indices,
	vector<int>& next_struct_indices,
	vector<Point3d>& structure,
	vector<Point3d>& next_structure,
	vector<Vec3b>& colors,
	vector<Vec3b>& next_colors
	)
{
	for (int i = 0; i < matched.size(); ++i)
	{
		int query_idx = matched[i].queryIdx;
		int train_idx = matched[i].trainIdx;

		int struct_idx = struct_indices[query_idx];
		if (struct_idx >= 0)
		{
			next_struct_indices[train_idx] = struct_idx;
			continue;
		}
		structure.push_back(next_structure[i]);
		colors.push_back(next_colors[i]);
		struct_indices[query_idx] = next_struct_indices[train_idx] = structure.size() - 1;
	}
}




struct ReprojectCost
{
    Point2d observation;

    ReprojectCost(Point2d& observation)
        : observation(observation)
    {
    }
    template <typename T>
    bool operator()(const T* const extrinsic, const T* const structure, T* residuals) const
    {
        residuals[0] = cos(extrinsic[0])*structure[0]-sin(extrinsic[0])*structure[1]+extrinsic[1] - T(observation.x);
        residuals[1] = sin(extrinsic[0])*structure[0]+cos(extrinsic[0])*structure[1]+extrinsic[2] - T(observation.y);

        return true;
    }
};

void ground_truth(Mat &result){
	ifstream csvPath ("ground_truth.txt");
	string line, sx, sy;
	double x, y;
	double x0 = 57.71386186, y0 = 11.94886661, dx = 30.9, dy = 30.9*cos(x0/180*3.1415);
	double resultResize = 15;
	double theta = 46*PI/180;
	// Mat result = Mat::zeros(resultSize, resultSize, CV_8UC3);

    // while (getline(csvPath, line))
    for(int i = 0; i < 1700; i++)
    {
    	getline(csvPath, line);
        stringstream liness(line);
        getline(liness, sx, ' ');
        getline(liness, sy, ' ');

        x = (stod(sx)-x0)*3600*dx;
        y = (stod(sy)-y0)*3600*dy;

		double xt = cos(theta)*x-sin(theta)*y;
		double yt = sin(theta)*x+cos(theta)*y;

		xt = xt * resultResize + resultSize/6;
		yt = yt * resultResize + resultSize/2;

		if (xt >= 0 && xt <= resultSize && yt >= 0 && yt <= resultSize){

			circle(result, Point(yt,xt), 1, Scalar (255,0,255), -1);
		}
    }
    // flip(result, result, 0	);
 //    namedWindow("result", WINDOW_NORMAL);
	// imshow("result", result);
	// waitKey(0);
}

cv::Mat module(int i, vector<vector<KP>> &keypoints_for_all, vector<vector<Vec3b>> &colors_for_all, vector<vector<int>> &correspond_struct_idx){
	Vec3b blue(255,255,0);
	Vec3b yellow(0,255,255);
	Vec3b orange(0,165,255);
	Vec3b orange2(0,0,255);

	// String img_path = "images/"+to_string(i)+".png";
	// String img_path = "tmp/img.png";
	// cv::Mat img = cv::imread(img_path);
	// img *= 2;
	// cv::imwrite("tmp/img1.png", img);
	// show_image(img);
	cv::Mat img = forwardDetectionORB(i);

	vector<KP> keypoints;
	vector<Vec3b> colors;
	ifstream csvPath ( "results/"+to_string(i)+".csv" );
	string line, x, y, label, X, Y, Z;
	int id;
	// Mat imgLast, imgNext, outImg;
    // Mat img = imread("result/"+to_string(i)+".png");
    while (getline(csvPath, line))
    {
        stringstream liness(line);
        getline(liness, x, ',');
        getline(liness, y, ',');
        getline(liness, label, ',');
        getline(liness, X, ',');
        getline(liness, Y, ',');
        getline(liness, Z, ',');

        // circle(img, Point (stoi(x),stoi(y)), 3, Scalar (0,0,0), -1);
        // cout << stod(Z) << endl;
        if(stod(Z)<1 && stod(Z)>0){
        	if(label == "blue"){
	            id = 0;
	            colors.push_back(blue);
	        }
	        if(label == "yellow"){
	            id = 1;
	            colors.push_back(yellow);
	        }
	        if(label == "orange"){
	            id = 2;
	            colors.push_back(orange);
	        }
	        if(label == "orange2"){
	            id = 3;
	            colors.push_back(orange2);
	        }
            Point3d pt(stod(X),stod(Z),1);
        	KP keypoint = {pt, id};
        	keypoints.push_back(keypoint);
        }

	}
	if(keypoints.size()<2){
		cout << i << ": too few keypoint!" << endl;
		return img;
	}
	keypoints_for_all.push_back(keypoints);
	colors_for_all.push_back(colors);

	vector<int> struct_idx;
	struct_idx.resize(keypoints.size(), -1);
	correspond_struct_idx.push_back(struct_idx);

	return img;
}

int main( int argc, char** argv )
{
	CNN("model", m_model);
	// string data_path = argv[1];
	// int start = stoi(argv[1]);
	// int end = stoi(argv[2]);
	int start = 1;
	int end = 226;
	Mat K(Matx33d(
		350.6847, 0, 332.4661,
		0, 350.0606, 163.7461,
		0, 0, 1));
	Vec3b blue(255,255,0);
	Vec3b yellow(0,255,255);
	Vec3b orange(0,165,255);
	Vec3b orange2(0,0,255);

	vector<Point3d> structure;
	vector<Vec3b> colors;
	vector<vector<int>> correspond_struct_idx;
	double resultResize = 140;

	vector<vector<KP>> keypoints_for_all;
	vector<vector<Vec3b>> colors_for_all;
	//vector<vector<DMatch>> matched_for_all;
	vector<Mat> affines;
	Mat affine = Mat::eye(3,3,CV_64F);
	affines.push_back(affine);

	for(int i = start; i < start+2; i++)
	{
		// clock_t currentTime = clock();

    	module(i, keypoints_for_all, colors_for_all, correspond_struct_idx);

    // 	double dur = (double)(clock()-currentTime);
		// cout << dur/1000 << endl;
	}

	init_structure(
		keypoints_for_all,
		colors_for_all,
		structure,
		colors,
		correspond_struct_idx,
		affines
		);

	int global_i = 0;
	for (int ii = start+2; ii < end+1; ii++)
	{
		// clock_t currentTime = clock();
		global_i++;
		cv::Mat img = module(ii, keypoints_for_all, colors_for_all, correspond_struct_idx);

		Mat affine, affine_tmp;
		vector<Point3d> p2, p2_tmp;
		vector<Vec3b> c1, c1_tmp;
		vector<DMatch> matched, matched_tmp;
		int next_img_id;
		double min_error1, min_error2;
		reconstruct(keypoints_for_all[global_i], keypoints_for_all[global_i+1], colors_for_all[global_i], colors_for_all[global_i+1], matched, c1, p2, affine, min_error1);

	    affine = affine*affines.back();
		affines.push_back(affine);

		vector<Point3d> next_structure;
		for(int i = 0; i < p2.size(); i++){
			next_structure.push_back(Point3d(Mat(affine.inv()*Mat(p2[i]))));
		}

		fusion_structure(
			matched,
			correspond_struct_idx[global_i],
			correspond_struct_idx[global_i+1],
			structure,
			next_structure,
			colors,
			c1
			);
		// cout << "from "<< i << " to " << next_img_id << endl;

		Mat result = Mat::zeros(resultSize, resultSize, CV_8UC3);
		vector<Point2d> path;
		ground_truth(result);
		for(int i = 0; i < structure.size(); i++){
			int x = int(structure[i].x * resultResize + resultSize/2);
			int y = int(structure[i].y * resultResize + resultSize/6);
			if (x >= 0 && x <= resultSize && y >= 0 && y <= resultSize){
				circle(result, Point (x,y), 3, colors[i], -1);
				// putText(result, to_string(i), Point(x,y), 1, 0.5, Scalar(255, 255, 255));
			}
		}

		for(int i = 0; i < affines.size(); i++){
			Mat camera_cor(Matx31d(0,0,1));
			camera_cor = affines[i].inv() * camera_cor;
			int x = int(camera_cor.at<double>(0,0) * resultResize + resultSize/2);
			int y = int(camera_cor.at<double>(1,0) * resultResize + resultSize/6);
			if (x >= 0 && x <= resultSize && y >= 0 && y <= resultSize){
				circle(result, Point (x,y), 2, Scalar (255,255,255), -1);
			}
			path.push_back(Point2d(x,y));
		}
		for(int i=0; i<path.size()-1; i++){
			line(result, path[i], path[i+1], Scalar (255,255,255), 1, 1, 0);
		}

		flip(result, result, 0);
		resize(img, img, cv::Size(), 2, 2);
		hconcat(img, result, result);
    namedWindow("result", WINDOW_NORMAL);
    cv::setWindowProperty("result", cv::WND_PROP_FULLSCREEN , cv::WINDOW_FULLSCREEN );
		imshow("result", result);
		waitKey(30);
		// imwrite("svo_results/"+to_string(global_i)+".png", result);

		// double dur = (double)(clock()-currentTime);
		// cout << dur/1000 << endl;
	}

	vector<int> count_same_structure;
	count_same_structure.resize(structure.size());
	for (int i = 0; i < correspond_struct_idx.size(); ++i){
		for (int j = 0; j < correspond_struct_idx[i].size(); ++j){
			// cout << correspond_struct_idx[i][j] << " ";
			for(int k = 0; k < structure.size(); k++){
				if(correspond_struct_idx[i][j] == k)
					count_same_structure[k]++;
			}
		}
		// cout << "\n";
	}


	// // bundle_adjustment
	// google::InitGoogleLogging(argv[0]);
	// vector<Mat> extrinsics;
	// for (size_t i = 0; i < affines.size(); ++i)
	// {
	//   Mat extrinsic(Matx31d(asin(affines[i].at<double>(1,0)), affines[i].at<double>(0,2), affines[i].at<double>(1,2)));
	//   extrinsics.push_back(extrinsic);
	// }
	// bundle_adjustment(extrinsics, correspond_struct_idx, keypoints_for_all, structure);

	// for (size_t i = 0; i < affines.size(); ++i)
	// {
	// 	double alpha = extrinsics[i].at<double>(0,0);
	// 	double tx = extrinsics[i].at<double>(1,0);
	// 	double ty = extrinsics[i].at<double>(2,0);
	// 	affines[i] = (Mat_<double>(3, 3) << cos(alpha), -sin(alpha), tx, sin(alpha), cos(alpha), ty, 0, 0, 1);
	// }

	Mat result = Mat::zeros(resultSize, resultSize, CV_8UC3);
	vector<Point2d> path;
	ground_truth(result);

	int count = 0;
	for(int i = 0; i < structure.size(); i++){
		// cout << count_same_structure[i] << endl;
		// cout << structure[i] << colors[i] << endl;
		if(count_same_structure[i] > rm_cone_threshold){
			count++;
			int x = int(structure[i].x * resultResize + resultSize/2);
			int y = int(structure[i].y * resultResize + resultSize/6);
			if (x >= 0 && x <= resultSize && y >= 0 && y <= resultSize){
				circle(result, Point (x,y), 3, colors[i], -1);
				// putText(result, to_string(i), Point(x,y), 1, 0.5, Scalar(255, 255, 255));
			}
		}
	}
	// cout << "Number of structure: " << count << endl;


	for(int i = 0; i < affines.size(); i++){
		Mat camera_cor(Matx31d(0,0,1));
		camera_cor = affines[i].inv() * camera_cor;
		// if(i>0)
		// 	cout << "heading change: " << acos(affines[i].at<double>(0,0))-acos(affines[i-1].at<double>(0,0)) << endl;
		int x = int(camera_cor.at<double>(0,0) * resultResize + resultSize/2);
		int y = int(camera_cor.at<double>(1,0) * resultResize + resultSize/6);
		if (x >= 0 && x <= resultSize && y >= 0 && y <= resultSize){
			circle(result, Point (x,y), 2, Scalar (255,255,255), -1);
		}
		path.push_back(Point2d(x,y));
	}
	for(int i=0; i<path.size()-1; i++){
		line(result, path[i], path[i+1], Scalar (255,255,255), 1, 1, 0);
	}

	flip(result, result, 0);
  namedWindow("result", WINDOW_NORMAL);
	imshow("result", result);
	waitKey(0);
	// imwrite("result/"+to_string(i)+".png", result);
}
