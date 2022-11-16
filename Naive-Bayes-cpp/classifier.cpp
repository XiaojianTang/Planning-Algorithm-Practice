#include "classifier.h"
#include <math.h>
#include <string>
#include <vector>

using Eigen::ArrayXd;
using std::string;
using std::vector;

#define M_PI 3.14159265358979323846

// Initializes GNB
GNB::GNB() {
  /**
   * TODO: Initialize GNB, if necessary. May depend on your implementation.
   */

  //初始化各个标签下数据得均值和方差[0,0,0,0]分别代表[s,d,s_dot,d_dot]，并初始化上一时刻值
  left_means = ArrayXd(4);
  left_means << 0,0,0,0;

  left_stds = ArrayXd(4);
  left_stds << 0,0,0,0;

  left_prior = 0;

  keep_means = ArrayXd(4);
  keep_means << 0,0,0,0;

  keep_stds = ArrayXd(4);
  keep_stds << 0,0,0,0;

  keep_prior = 0;

  right_means = ArrayXd(4);
  right_means << 0,0,0,0;

  right_stds = ArrayXd(4);
  right_stds << 0,0,0,0;

  right_prior = 0;
}

GNB::~GNB() {}

void GNB::train(const vector<vector<double>> &data, 
                const vector<string> &labels) {
  /**
   * Trains the classifier with N data points and labels.
   * @param data - array of N observations
   *   - Each observation is a tuple with 4 values: s, d, s_dot and d_dot.
   *   - Example : [[3.5, 0.1, 5.9, -0.02],
   *                [8.0, -0.3, 3.0, 2.2],
   *                 ...
   *                ]
   * @param labels - array of N labels
   *   - Each label is one of "left", "keep", or "right".
   *
   * TODO: Implement the training function for your classifier.
   * 
   */

  //计算均值

  //初始化各标签数量
  float left_size =0;
  float keep_size = 0;
  float right_size = 0; 

  //循环整个labels列表，分别提取出不同标签下得数据，并计算其均值
  //注意，因为data得数据类型为vector，需将其转化为Eigen库里得ArrayXd,需使用ArrayXd::Map()

  //定义一个ArrayXd变量，用于临时储存将从data中提取并转化为ArrayXd得数据
  ArrayXd data_point;

  for(int i=0;i<labels.size();i++){

    //转化数据类型
    data_point = ArrayXd::Map(data[i].data(),data[i].size());    
    
    if (labels[i] == "left") {      
      left_means += data_point;
      left_size += 1;
    } else if (labels[i] == "keep") {
      keep_means += data_point;
      keep_size += 1;
    } else if (labels[i] == "right") {
      right_means += data_point;
      right_size += 1;
    }
  }

    //计算均值
  left_means = left_means/left_size;
  keep_means = keep_means/keep_size;
  right_means = right_means/right_size;

    //计算标准差
  for(int i=0;i<labels.size();i++){

    //转化数据类型
    data_point = ArrayXd::Map(data[i].data(),data[i].size());

    if(labels[i] == "left"){
      left_stds += (data_point - left_means) *  (data_point - left_means);
    }
    else if(labels[i] == "keep"){
      keep_stds += (data_point - keep_means) *  (data_point - keep_means);
    }
    else if(labels[i] == "right"){
      right_stds += (data_point - right_means) *  (data_point - right_means);
    }
  }

  //计算标准差
  left_stds = (left_stds/left_size).sqrt();
  keep_stds = (keep_stds/keep_size).sqrt();
  right_stds = (right_stds/keep_size).sqrt();

  //计算各个标签得比例（各标签得先验概率）
  left_prior = left_size/labels.size();
  keep_prior = keep_size/labels.size();
  right_prior = right_size/labels.size();
    
}

string GNB::predict(const vector<double> &sample) {
  /**
   * Once trained, this method is called and expected to return 
   *   a predicted behavior for the given observation.
   * @param observation - a 4 tuple with s, d, s_dot, d_dot.
   *   - Example: [3.5, 0.1, 8.5, -0.2]
   * @output A label representing the best guess of the classifier. Can
   *   be one of "left", "keep" or "right".
   *
   * TODO: Complete this function to return your classifier's prediction
   */
  double left_p = 1.0;
  double keep_p = 1.0;
  double right_p = 1.0;

  //依次计算[s,d,s_dot,d_dot]得条件概率
  for(int i=0; i<4;i++){
    left_p *= (1.0/sqrt(2.0 * M_PI * pow(left_stds[i], 2))) 
            * exp(-0.5*pow(sample[i] - left_means[i], 2)/pow(left_stds[i], 2));
    keep_p *= (1.0/sqrt(2.0 * M_PI * pow(keep_stds[i], 2)))
            * exp(-0.5*pow(sample[i] - keep_means[i], 2)/pow(keep_stds[i], 2));
    right_p *= (1.0/sqrt(2.0 * M_PI * pow(right_stds[i], 2))) 
            * exp(-0.5*pow(sample[i] - right_means[i], 2)/pow(right_stds[i], 2));
    }

    left_p = left_p * left_prior;
    keep_p = keep_p * keep_prior;
    right_p = right_p * right_prior;

    //取概率最大
    double probs[3] = {left_p,keep_p,right_p};
    double max = left_p;
    double max_index = 0;

    for(int i=0; i<3; i++){
      if(probs[i] > max){
        max = probs[i];
        max_index = i;
      }
    }  

  return this -> possible_labels[max_index];
}