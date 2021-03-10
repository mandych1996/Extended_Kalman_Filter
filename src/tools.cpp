#include "tools.h"
#include <iostream>

using namespace std;
using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
   VectorXd rmse(4);
   rmse << 0, 0, 0, 0;

   if(estimations.size() != ground_truth.size() || estimations.size() == 0){
      cout << "Invalid data -- estimation / ground_truth" << endl;
      return rmse;
   }
   // accumulate squared residuals
   for(int i =0; i < estimations.size(); i++){
      VectorXd residual = estimations[i] - ground_truth[i];
      residual = residual.array()* residual.array();
      rmse += residual;
   }
   // calculate the mean
   rmse = rmse / estimations.size();

   // calculate the squared root
   rmse = rmse.array().sqrt();

   return rmse;
}

MatrixXd Tools::CalculateJacobian(const VectorXd& x_state) {
   MatrixXd Hj(3, 4);

   float px = x_state(0);
   float py = x_state(1);
   float vx = x_state(2);
   float vy = x_state(3);
 
   // Avoid division by zero
   float min_value = 0.0001;
   if(fabs(px) < min_value && fabs(py) < min_value){
      px = min_value;
      py = min_value;
   }
  // Avoid repeated calculation
   float c1 = px*px + py*py;
   float c2 = sqrt(c1);
   float c3 = c1*c2;

  // Avoid division by zero
   if(fabs(c1) < min_value){
      cout << "CalculateJacobian () - Error - Division by Zero" << endl;
      return Hj;
   }
   // Jacobian Matrix
   Hj << px/c2, py/c2, 0, 0,
         -py/c2, px/c1, 0, 0,
         py*(vx*py-vy*px)/c3, px*(px*vy-py*vx)/c3, px/c2, py/c2;
   
   return Hj;
}
