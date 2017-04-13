#ifndef KALMAN_FILTER_H_
#define KALMAN_FILTER_H_
#include "Eigen/Dense"
namespace KF
{
	void Update(Eigen::MatrixXd &H_,
		Eigen::VectorXd &x_,
		Eigen::VectorXd z_,
		Eigen::VectorXd z_pred,
		Eigen::MatrixXd &P_,
		Eigen::MatrixXd &R_,
		Eigen::MatrixXd &I);
	void Predict(Eigen::VectorXd &x_,
		Eigen::MatrixXd &F_,
		Eigen::MatrixXd &P_,
		Eigen::MatrixXd &Q_);
}
#endif KALMAN_FILTER_H_ 
