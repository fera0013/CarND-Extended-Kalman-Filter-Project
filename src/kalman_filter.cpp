#include "kalman_filter.h"

void KF::Update(Eigen::MatrixXd & H_, Eigen::VectorXd & x_, Eigen::VectorXd z_, Eigen::VectorXd z_pred, Eigen::MatrixXd & P_, Eigen::MatrixXd & R_, Eigen::MatrixXd & I)
{
	Eigen::VectorXd y = z_ - z_pred;
	Eigen::MatrixXd Ht = H_.transpose();
	Eigen::MatrixXd S = H_ * P_ * Ht + R_;
	Eigen::MatrixXd Si = S.inverse();
	Eigen::MatrixXd PHt = P_ * Ht;
	Eigen::MatrixXd K = PHt * Si;

	//new estimate
	x_ = x_ + (K * y);
	long x_size = x_.size();
	P_ = (I - K * H_) * P_;
}

void KF::Predict(Eigen::VectorXd & x_, Eigen::MatrixXd & F_, Eigen::MatrixXd & P_, Eigen::MatrixXd & Q_)
{
	x_ = F_ * x_;
	auto Ft = F_.transpose();
	P_ = F_ * P_ * Ft + Q_;
}
