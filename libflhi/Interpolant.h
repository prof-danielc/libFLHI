#ifndef INTERPOLANT_H_INCLUDED
#define INTERPOLANT_H_INCLUDED

#include <vector>
#include <Eigen/Dense>
#include <Eigen/StdVector>
#include <set>
#include <cppoptlib/boundedproblem.h>
#include "GridData.h"

namespace FLHI
{

struct CubicData
{
	Eigen::MatrixXd f0, f1, df0, df1;

	CubicData(Eigen::DenseIndex inputDimensionCount, Eigen::DenseIndex outputDimensionCount)
		:
		f0(inputDimensionCount, outputDimensionCount),
		f1(inputDimensionCount, outputDimensionCount),
		df0(inputDimensionCount, outputDimensionCount),
		df1(inputDimensionCount, outputDimensionCount)
	{
	}
};

class Interpolant
{
public:
	enum KFTYPE
	{
		KF_NEARESTNEIGHBOR,
		KF_LINEAR,
		KF_CUBIC,
		KF_LANCZOS,
		KF_SPLINE
	};

	GridData grid;

	// For cubic interpolation
	// This is realized at a memory trade-off for efficiency
	// dxo and cubicData increase memory cost by at least 3x "grid.pointCount", however, this allows membershipEvaluate to be blazing fast
	std::vector< Eigen::MatrixXd, Eigen::aligned_allocator<Eigen::MatrixXd> > dxo;
	std::vector< std::vector<CubicData> > cubicData;

	std::vector<KFTYPE> kernelFunctionOptions;

	// for reverse interpolation
	Eigen::VectorXd ub;
	Eigen::VectorXd lb;
	Eigen::VectorXd xiInitial;

	// TODO: where was I going with this? maybe the intervaltree?
	//std::multimap< std::pair<Eigen::VectorXd, Eigen::VectorXd> >

public:
	Interpolant(const Eigen::Ref<const Eigen::MatrixXd>& pointVectorXi, const Eigen::Ref<const Eigen::MatrixXd>& pointVectorXo, KFTYPE kernelFunction = KF_LINEAR);
	Interpolant(const Eigen::Ref<const Eigen::MatrixXd>& pointVectorXi, const Eigen::Ref<const Eigen::MatrixXd>& pointVectorXo, const std::vector<KFTYPE> &kernelFunctions);

	void buildCubicData();
	void buildDifferencesTable();
	void centralDifference(Eigen::DenseIndex pointIndex, Eigen::DenseIndex inputIndex);

	Eigen::VectorXd interpolateBounded(const Eigen::VectorXd &xi);
	Eigen::VectorXd interpolate(const Eigen::VectorXd &xi, bool checkBounds=true);
	Eigen::VectorXd interpolateRegion(const Region& region, const Eigen::VectorXd &xi);

	std::vector< Eigen::VectorXd, Eigen::aligned_allocator<Eigen::VectorXd> > interpolateInverse(const Eigen::VectorXd &xo);

	double tnorm(const Eigen::VectorXd& kernelMembership);

	double deffuzification(const Eigen::MatrixXd &membership, const Eigen::MatrixXd &xoRegion, Eigen::DenseIndex xoIndex);

	double membershipEvaluate(Eigen::DenseIndex vertexPointIndex, double x, Eigen::DenseIndex inputIndex, Eigen::DenseIndex outputIndex);
	double kernelNearestNeighbor(double x);
	double kernelLinear(double x);
	double kernelCubic(double x, double f0, double f1, double df0, double df1);
	double kernelLanczos(double x);
	double kernelSpline(double x);
};

// Objective function class for inverse interpolation minimization problem
class ObjectiveFunction : public cppoptlib::BoundedProblem<double>
{
private:
	// TODO: make everything const correct
	//const Interpolant& flhi;
	//const Region& region;
	//const Eigen::VectorXd& xoExpected;
	Interpolant& flhi;
	Region& region;
	const Eigen::VectorXd& xoExpected;

public:
	using Superclass = cppoptlib::BoundedProblem<double>;
	using typename Superclass::TVector;
	using typename Superclass::THessian;

	ObjectiveFunction(Interpolant &flhi, Region &region, const Eigen::VectorXd &xoExpected) : Superclass(xoExpected.size()), flhi(flhi), region(region), xoExpected(xoExpected) {}

	double value(const TVector &xi)
	{
		// NOTE: in theory cppoptlib::Problem::TVector and Eigen::VectorXd should be the same
		Eigen::VectorXd xo = flhi.interpolateRegion(region, xi);

		// sum of squared residuals
		return (xo - xoExpected).array().square().sum();
	}

	// TODO: add gradient
	//void gradient(const TVector &xi, TVector &grad)
};

}

#endif // INTERPOLANT_H_INCLUDED
