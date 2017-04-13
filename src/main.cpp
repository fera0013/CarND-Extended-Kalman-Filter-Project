#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>
#include <stdlib.h>
#include "Eigen/Dense"
#include "ground_truth_package.h"
#include "measurement_package.h"
#include <iostream>
#include "tools.h"
#include "kalman_filter.h"

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

void check_arguments(int argc, char* argv[]) {
	string usage_instructions = "Usage instructions: ";
	usage_instructions += argv[0];
	usage_instructions += " path/to/input.txt output.txt";

	bool has_valid_args = false;

	// make sure the user has provided input and output files
	if (argc == 1) {
		cerr << usage_instructions << endl;
	}
	else if (argc == 2) {
		cerr << "Please include an output file.\n" << usage_instructions << endl;
	}
	else if (argc == 3) {
		has_valid_args = true;
	}
	else if (argc > 3) {
		cerr << "Too many arguments.\n" << usage_instructions << endl;
	}

	if (!has_valid_args) {
		exit(EXIT_FAILURE);
	}
}

void check_files(ifstream& in_file, string& in_name,
	ofstream& out_file, string& out_name) {
	if (!in_file.is_open()) {
		cerr << "Cannot open input file: " << in_name << endl;
		exit(EXIT_FAILURE);
	}

	if (!out_file.is_open()) {
		cerr << "Cannot open output file: " << out_name << endl;
		exit(EXIT_FAILURE);
	}
}


VectorXd Zpred(MeasurementPackage &measurement_pack, Eigen::MatrixXd &H_, Tools &tools, Eigen::VectorXd &x_, Eigen::MatrixXd &R_, Eigen::MatrixXd &R_radar_, Eigen::MatrixXd &H_laser_, Eigen::MatrixXd &R_laser_)
{
	VectorXd z_pred;
	if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
		// Radar updates
		H_ = tools.CalculateJacobian(x_);
		R_ = R_radar_;
		auto rho = sqrt(x_[0] * x_[0] + x_[1] * x_[1]);
		z_pred = VectorXd(3);
		z_pred << rho,
			atan2(x_[1], x_[0]),
			(x_[0] * x_[2] + x_[1] * x_[3]) / rho;
	}
	else {
		H_ = H_laser_;
		R_ = R_laser_;
		z_pred = H_ * x_;
	}
	return z_pred;
}

int main(int argc, char* argv[]) {
	check_arguments(argc, argv);

	vector<MeasurementPackage> measurement_pack_list;
	vector<GroundTruthPackage> gt_pack_list;

	string in_file_name_ = argv[1];
	ifstream in_file_(in_file_name_.c_str(), ifstream::in);

	string out_file_name_ = argv[2];
	ofstream out_file_(out_file_name_.c_str(), ofstream::out);

	check_files(in_file_, in_file_name_, out_file_, out_file_name_);
	Tools tools;
	tools.ReadMeasurementPacks(in_file_, measurement_pack_list, gt_pack_list);

	bool is_initialized_ = false;


	// used to compute the RMSE later
	vector<VectorXd> estimations;
	vector<VectorXd> ground_truth;

	long long previous_timestamp_;

	// initializing matrices
	MatrixXd R_laser_ = MatrixXd(2, 2);
	R_laser_ << 0.0225, 0,
		0, 0.0225;
	MatrixXd R_radar_ = MatrixXd(3, 3);
	R_radar_ << 0.09, 0, 0,
		0, 0.0009, 0,
		0, 0, 0.09;
	MatrixXd H_laser_ = MatrixXd(2, 4);
	H_laser_ << 1, 0, 0, 0,
		0, 1, 0, 0;
	MatrixXd Hj_ = MatrixXd(3, 4);
	MatrixXd F_ = MatrixXd(4, 4);
	F_ << 1, 0, 1, 0,
		0, 1, 0, 1,
		0, 0, 1, 0,
		0, 0, 0, 1;
	MatrixXd P_ = MatrixXd(4, 4);
	P_ << 1, 0, 0, 0,
		0, 1, 0, 0,
		0, 0, 1000, 0,
		0, 0, 0, 1000;
	float noise_ax = 9;
	float noise_ay = 9;
	VectorXd x_ = VectorXd(4);
	x_ << 1, 1, 1, 1;
	MatrixXd I = MatrixXd::Identity(x_.size(), x_.size());

	size_t N = measurement_pack_list.size();
	for (size_t k = 0; k < N; ++k) {
		auto measurement_pack = measurement_pack_list[k];
		auto gt_pack = gt_pack_list[k];
		if (!is_initialized_) {
			if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
				/**
				Convert radar from polar to cartesian coordinates and initialize state.
				*/
				auto rho = measurement_pack.raw_measurements_[0];
				auto phi = measurement_pack.raw_measurements_[1];
				x_ << rho*cos(phi), rho*sin(phi), 0, 0;
			}
			else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) {
				/**
				Initialize state.
				*/
				x_ << measurement_pack.raw_measurements_[0], measurement_pack.raw_measurements_[1], 0, 0;
			}

			// done initializing, no need to predict or update
			is_initialized_ = true;
			previous_timestamp_ = measurement_pack.timestamp_;
			continue;
		}
		//Prevent xpos and ypos from getting too small
		if (fabs(x_[0]) < 0.0001 &&
			fabs(x_[1])< 0.0001)
		{
			x_[0] = 0.0001;
			x_[1] = 0.0001;
		}
		/*****************************************************************************
		*  Prediction
		****************************************************************************/
		float dt = (measurement_pack.timestamp_ - previous_timestamp_) / 1000000.0;	//dt - expressed in seconds
		previous_timestamp_ = measurement_pack.timestamp_;
		float dt_2 = dt * dt;
		float dt_3 = dt_2 * dt;
		float dt_4 = dt_3 * dt;
		// Update the state transition matrix F according to the new elapsed time.
		F_(0, 2) = dt;
		F_(1, 3) = dt;
		MatrixXd Q_ = MatrixXd(4, 4);
		Q_ << dt_4 / 4 * noise_ax, 0, dt_3 / 2 * noise_ax, 0,
			0, dt_4 / 4 * noise_ay, 0, dt_3 / 2 * noise_ay,
			dt_3 / 2 * noise_ax, 0, dt_2*noise_ax, 0,
			0, dt_3 / 2 * noise_ay, 0, dt_2*noise_ay;
		KF::Predict(x_, F_, P_, Q_);
		MatrixXd H_;
		MatrixXd R_;
		VectorXd z_pred = Zpred(measurement_pack, H_, tools, x_, R_, R_radar_, H_laser_, R_laser_);
		KF::Update(H_, x_, measurement_pack.raw_measurements_, z_pred, P_, R_, I);

		cout << "x_ = " << x_ << endl;
		cout << "P_ = " << P_ << endl;

		tools.WriteToOutput(out_file_, x_, measurement_pack, gt_pack);



		estimations.push_back(x_);
		ground_truth.push_back(gt_pack_list[k].gt_values_);
	}

	// compute the accuracy (RMSE)
	cout << "Accuracy - RMSE:" << endl << tools.CalculateRMSE(estimations, ground_truth) << endl;

	// close files
	if (out_file_.is_open()) {
		out_file_.close();
	}

	if (in_file_.is_open()) {
		in_file_.close();
	}

	return 0;
}

bool is_initialized_;

