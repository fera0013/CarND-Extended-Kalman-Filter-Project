#include <iostream>
#include "tools.h"
#include <fstream>
#include <iostream>
#include <sstream>


using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

void Tools::ReadMeasurementPacks(std::ifstream & in_file_, std::vector<MeasurementPackage>& measurement_pack_list, std::vector<GroundTruthPackage>& gt_pack_list)
{
	std::string line;

	// prep the measurement packages (each line represents a measurement at a
	// timestamp)
	while (getline(in_file_, line)) {

		std::string sensor_type;
		MeasurementPackage meas_package;
		GroundTruthPackage gt_package;
		std::istringstream iss(line);
		long long timestamp;
		// reads first element from the current line
		iss >> sensor_type;
		if (sensor_type.compare("L") == 0) {
			// LASER MEASUREMENT

			// read measurements at this timestamp
			meas_package.sensor_type_ = MeasurementPackage::LASER;
			meas_package.raw_measurements_ = VectorXd(2);
			float x;
			float y;
			iss >> x;
			iss >> y;
			meas_package.raw_measurements_ << x, y;
			iss >> timestamp;
			meas_package.timestamp_ = timestamp;
			measurement_pack_list.push_back(meas_package);
		}
		else if (sensor_type.compare("R") == 0) {
			// RADAR MEASUREMENT

			// read measurements at this timestamp
			meas_package.sensor_type_ = MeasurementPackage::RADAR;
			meas_package.raw_measurements_ = VectorXd(3);
			float ro;
			float phi;
			float ro_dot;
			iss >> ro;
			iss >> phi;
			iss >> ro_dot;
			meas_package.raw_measurements_ << ro, phi, ro_dot;
			iss >> timestamp;
			meas_package.timestamp_ = timestamp;
			measurement_pack_list.push_back(meas_package);
		}

		// read ground truth data to compare later
		float x_gt;
		float y_gt;
		float vx_gt;
		float vy_gt;
		iss >> x_gt;
		iss >> y_gt;
		iss >> vx_gt;
		iss >> vy_gt;
		gt_package.gt_values_ = VectorXd(4);
		gt_package.gt_values_ << x_gt, y_gt, vx_gt, vy_gt;
		gt_pack_list.push_back(gt_package);
	}
}

void Tools::WriteToOutput(std::ofstream & out_file_, const VectorXd x_, const MeasurementPackage & measurement_pack, const GroundTruthPackage & gt_pack)
{
	// output the estimation
	out_file_ << x_(0) << "\t";
	out_file_ << x_(1) << "\t";
	out_file_ << x_(2) << "\t";
	out_file_ << x_(3) << "\t";

	// output the measurements
	if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) {
		// output the estimation
		out_file_ << measurement_pack.raw_measurements_(0) << "\t";
		out_file_ << measurement_pack.raw_measurements_(1) << "\t";
	}
	else if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
		// output the estimation in the cartesian coordinates
		float ro = measurement_pack.raw_measurements_(0);
		float phi = measurement_pack.raw_measurements_(1);
		out_file_ << ro * cos(phi) << "\t"; // p1_meas
		out_file_ << ro * sin(phi) << "\t"; // ps_meas
	}

	// output the ground truth packages
	out_file_ << gt_pack.gt_values_(0) << "\t";
	out_file_ << gt_pack.gt_values_(1) << "\t";
	out_file_ << gt_pack.gt_values_(2) << "\t";
	out_file_ << gt_pack.gt_values_(3) << "\n";

}


VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
	const vector<VectorXd> &ground_truth) {
	/**
	TODO:
	* Calculate the RMSE here.
	*/
	VectorXd rmse(4);
	rmse << 0, 0, 0, 0;

	// check the validity of the following inputs:
	//  * the estimation vector size should not be zero
	//  * the estimation vector size should equal ground truth vector size
	if (estimations.size() != ground_truth.size()
		|| estimations.size() == 0) {
		return rmse;
	}

	//accumulate squared residuals
	for (unsigned int i = 0; i < estimations.size(); ++i) {

		VectorXd residual = estimations[i] - ground_truth[i];

		//coefficient-wise multiplication
		residual = residual.array()*residual.array();
		rmse += residual;
	}

	//calculate the mean
	rmse = rmse / estimations.size();

	//calculate the squared root
	rmse = rmse.array().sqrt();

	//return the result
	return rmse;
}

MatrixXd Tools::CalculateJacobian(const VectorXd& x_state) {
	/**
	TODO:
	* Calculate a Jacobian here.
	*/
	MatrixXd Hj(3, 4);
	//recover state parameters
	float px = x_state(0);
	float py = x_state(1);
	float vx = x_state(2);
	float vy = x_state(3);

	//pre-compute a set of terms to avoid repeated calculation
	float c1 = px*px + py*py;

	//check division by zero
	if (fabs(c1) < 0.0001) {
		// return Hj;
		c1 = 0.0001;
	}

	float c2 = sqrt(c1);
	float c3 = (c1*c2);



	//compute the Jacobian matrix
	Hj << (px / c2), (py / c2), 0, 0,
		-(py / c1), (px / c1), 0, 0,
		py*(vx*py - vy*px) / c3, px*(px*vy - py*vx) / c3, px / c2, py / c2;

	return Hj;
}
