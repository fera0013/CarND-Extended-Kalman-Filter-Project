#ifndef TOOLS_H_
#define TOOLS_H_
#include <vector>
#include "Eigen/Dense"
#include "ground_truth_package.h"
#include "measurement_package.h"

class Tools {
public:
	/**
	* Constructor.
	*/
	Tools();

	/**
	* Destructor.
	*/
	virtual ~Tools();

	/**
	* A helper method to calculate RMSE.
	*/
	Eigen::VectorXd CalculateRMSE(const std::vector<Eigen::VectorXd> &estimations, const std::vector<Eigen::VectorXd> &ground_truth);

	/**
	* A helper method to calculate Jacobians.
	*/
	Eigen::MatrixXd CalculateJacobian(const Eigen::VectorXd& x_state);
	void ReadMeasurementPacks(std::ifstream &in_file_,
		std::vector<MeasurementPackage>& measurement_pack_list,
		std::vector<GroundTruthPackage>& gt_pack_list);

	void WriteToOutput(std::ofstream &out_file_,
		const Eigen::VectorXd x_,
		const MeasurementPackage &measurement_pack,
		const GroundTruthPackage &gt_pack);

};

#endif /* TOOLS_H_ */
