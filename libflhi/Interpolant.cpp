#include <stdexcept>
#include <cppoptlib/solver/lbfgsbsolver.h>
#include "Interpolant.h"

#include <iostream>
using namespace std;

namespace FLHI
{

Interpolant::Interpolant(const Eigen::Ref<const Eigen::MatrixXd>& pointVectorXi, const Eigen::Ref<const Eigen::MatrixXd>& pointVectorXo, KFTYPE kernelFunction)
	:
	Interpolant(pointVectorXi, pointVectorXo, std::vector<KFTYPE>(1, kernelFunction))
{
}

Interpolant::Interpolant(const Eigen::Ref<const Eigen::MatrixXd>& pointVectorXi, const Eigen::Ref<const Eigen::MatrixXd>& pointVectorXo, const std::vector<KFTYPE>& kernelFunctions)
	:
	grid(pointVectorXi, pointVectorXo),
	kernelFunctionOptions(kernelFunctions),
	ub(Eigen::VectorXd::Ones(grid.outputDimensionCount)),
	lb(Eigen::VectorXd::Zero(grid.outputDimensionCount)),
	xiInitial((grid.xiMaximum - grid.xiMinimum)/2)
{
	// If a single kernel option was passed, duplicate it for all input dimensions
	if (kernelFunctionOptions.size() == 1)
		kernelFunctionOptions.insert(kernelFunctionOptions.end(), grid.inputDimensionCount-1, kernelFunctionOptions[0]);

	// Eigen might define Eigen::DenseIndex to be an integer, however, inputDimensionCount is zero or positive anyways.
	if(static_cast<std::size_t>(grid.inputDimensionCount) != kernelFunctionOptions.size())
		throw std::invalid_argument(std::string("INTERPOLANT::INTERPOLANT kernelFunctions input must have size one or the same size as input dimensions."));

	// If cubic kernel function was selected, build the output differences table dxo
	for (std::size_t i = 0; i < kernelFunctionOptions.size(); i++)
	{
		if (kernelFunctionOptions[i] == KF_CUBIC)
		{
			buildDifferencesTable();
			buildCubicData();
			break;
		}
	}
}

void Interpolant::buildCubicData()
{
	cubicData.reserve(grid.pointCount);

	for (Eigen::DenseIndex pointIndex = 0; pointIndex < grid.pointCount; pointIndex++)
	{
		CubicData left(grid.inputDimensionCount, grid.outputDimensionCount); // when point is at origin vertex
		CubicData right(grid.inputDimensionCount, grid.outputDimensionCount); // when point is at unit vertex

		for (Eigen::DenseIndex inputIndex = 0; inputIndex < grid.inputDimensionCount; inputIndex++)
		{
			Eigen::DenseIndex hypercubeDimensionShiftPos = 1 << inputIndex;

			// Find nearest points after this point is shifted positively and negatively on an input dimension
			Eigen::DenseIndex pointIndexShift = grid.hypercubeShiftIndices[hypercubeDimensionShiftPos];

			Eigen::DenseIndex rightPointIndex = pointIndex + pointIndexShift;
			Eigen::DenseIndex leftPointIndex = pointIndex - pointIndexShift;

			// Keep it inside grid bounds
			if (rightPointIndex >= grid.pointCount)
				rightPointIndex = 0;

			if (leftPointIndex < 0)
				leftPointIndex = 0;

			// Sanity check on Eigen's dense index, if this fails then the if condition above will not work
			assert(static_cast<Eigen::DenseIndex>(-1) < 0);

			for (Eigen::DenseIndex outputIndex = 0; outputIndex < grid.outputDimensionCount; outputIndex++)
			{
				left.f0(inputIndex, outputIndex) = grid.dataXo.coeff(pointIndex, outputIndex);
				left.df0(inputIndex, outputIndex) = dxo[pointIndex].coeff(inputIndex, outputIndex);
				left.f1(inputIndex, outputIndex) = grid.dataXo.coeff(rightPointIndex, outputIndex);
				left.df1(inputIndex, outputIndex) = dxo[rightPointIndex].coeff(inputIndex, outputIndex);

				right.f0(inputIndex, outputIndex) = grid.dataXo.coeff(leftPointIndex, outputIndex);
				right.df0(inputIndex, outputIndex) = dxo[leftPointIndex].coeff(inputIndex, outputIndex);
				right.f1(inputIndex, outputIndex) = grid.dataXo.coeff(pointIndex, outputIndex);
				right.df1(inputIndex, outputIndex) = dxo[pointIndex].coeff(inputIndex, outputIndex);
			}
		}

		// NOTE: I tried a lot of fancy shit and this is as simple and fast as it gets (so far...)
		std::vector<CubicData> cubicVertices;
		cubicVertices.push_back(std::move(left));
		cubicVertices.push_back(std::move(right));

		cubicData.push_back(std::move(cubicVertices));
	}
}

void Interpolant::buildDifferencesTable()
{
	dxo.reserve(grid.pointCount);

	for (Eigen::DenseIndex pointIndex = 0; pointIndex < grid.pointCount; pointIndex++)
	{
		// This should perform a move
		dxo.push_back(Eigen::MatrixXd(grid.inputDimensionCount, grid.outputDimensionCount));

		// Calculate central differences for this point
		for (Eigen::DenseIndex inputIndex = 0; inputIndex < grid.inputDimensionCount; inputIndex++)
			centralDifference(pointIndex, inputIndex);
	}
}

void Interpolant::centralDifference(Eigen::DenseIndex pointIndex, Eigen::DenseIndex inputIndex)
{
	Eigen::DenseIndex hypercubeDimensionShiftPos = 1 << inputIndex;

	// Find nearest points after this point is shifted positively and negatively on an input dimension
	Eigen::DenseIndex pointIndexShift = grid.hypercubeShiftIndices[hypercubeDimensionShiftPos];

	Eigen::DenseIndex rightPointIndex = pointIndex + pointIndexShift;
	Eigen::DenseIndex leftPointIndex = pointIndex - pointIndexShift;

	// Keep it inside grid bounds
	if (rightPointIndex >= grid.pointCount)
		rightPointIndex = 0;

	if (leftPointIndex < 0)
		leftPointIndex = 0;

	// Sanity check on Eigen's dense index, if this fails then the if condition above will not work
	assert(static_cast<Eigen::DenseIndex>(-1) < 0);

	// Calculate central difference on this input dimension for this point, for all output dimensions
	for (Eigen::DenseIndex outputIndex = 0; outputIndex < grid.outputDimensionCount; outputIndex++)
	{
		double xoright = grid.dataXo.coeff(rightPointIndex, outputIndex);
		double xoleft = grid.dataXo.coeff(leftPointIndex, outputIndex);

		// Interpolation occurs at an unitary space which isn't dependent on step size, there is no need to take step size into account here
		dxo[pointIndex](inputIndex, outputIndex) = (xoright - xoleft) / 2;
	}
}

std::vector< Eigen::VectorXd, Eigen::aligned_allocator<Eigen::VectorXd> > Interpolant::interpolateInverse(const Eigen::VectorXd &xo)
{
	std::vector< Eigen::VectorXd, Eigen::aligned_allocator<Eigen::VectorXd> > xiSet;

	// TODO: currently a simple search is performed on all regions. this is *SLOW*.
	// This could be heavily optimized by using a kd-tree or an interval tree
	for (unsigned int i = 0; i < grid.regionCount; i++)
	{
		// TODO: this should be a const Region
		Region& region = grid.regions[i];

		bool isInUpperBound = (xo.array() <= region.xoMaximum.array()).all();
		bool isInLowerBound = (xo.array() >= region.xoMinimum.array()).all();
		bool isInRegion = isInUpperBound && isInLowerBound;

		if (isInRegion)
		{
			// Create objective function with upper and lower bounds in unitary hypercube space
			ObjectiveFunction fobj(*this, region, xo);
			fobj.setBoxConstraint(lb, ub);

			// Create the L-BFGS-B solver and minimize
			cppoptlib::LbfgsbSolver<ObjectiveFunction> solver;
			Eigen::VectorXd xiFound = xiInitial;
			solver.minimize(fobj, xiFound);

			cout << "Inverse Solution in hypercube coordinates: " << xiFound.transpose() << endl;

			Eigen::VectorXd xi = grid.hypercubeCoordinateToSpatialCoordinate(xiFound, region.pointsIndex[0]);
			cout << "Inverse Solution in spatial coordinates: " << xi.transpose() << endl;
			cout << "Inverse Solution evaluation: " << fobj(xiFound) << endl;

			// Add this solution to the solution set
			xiSet.push_back(xi);
		}
	}

	// NOTE: relies on return value optimization (RVO)
	return xiSet;
}

Eigen::VectorXd Interpolant::interpolateBounded(const Eigen::VectorXd &xi)
{
	if (xi.size() != grid.inputDimensionCount)
		throw std::invalid_argument(std::string("INTERPOLANT::INTERPOLATE point dimension count must match the input dimension count of internal grid."));

	Eigen::VectorXd boundedPoint = grid.limitPointToBounds(xi);

	Eigen::VectorXd xiHypercube(grid.inputDimensionCount);
	const Region &region = grid.findRegion(boundedPoint, xiHypercube);

	return interpolateRegion(region, xiHypercube);
}

Eigen::VectorXd Interpolant::interpolate(const Eigen::VectorXd &xi, bool checkBounds)
{
	if(xi.size() != grid.inputDimensionCount)
		throw std::invalid_argument(std::string("INTERPOLANT::INTERPOLATE point dimension count must match the input dimension count of internal grid."));

	if(checkBounds && grid.isPointBeyondBounds(xi))
		throw std::invalid_argument(std::string("INTERPOLANT::INTERPOLATE point is out of bounds of internal data. Consider using interpolateBounded."));

	Eigen::VectorXd xiHypercube(grid.inputDimensionCount);
	const Region &region = grid.findRegion(xi, xiHypercube);

	return interpolateRegion(region, xiHypercube);
}

Eigen::VectorXd Interpolant::interpolateRegion(const Region& region, const Eigen::VectorXd &xi)
{
	// TODO: sanity check xi for range [0, 1] ?
	// TODO: last point I was working on

	Eigen::VectorXd xo(grid.outputDimensionCount);
	Eigen::MatrixXd membership(grid.outputDimensionCount, grid.pointCountInRegion);

	Eigen::MatrixXd xoRegion(grid.outputDimensionCount, grid.pointCountInRegion);

	for (Eigen::DenseIndex xoIndex = 0; xoIndex < grid.outputDimensionCount; xoIndex++)
	{
		for (Eigen::DenseIndex pointIndex = 0; pointIndex < grid.pointCountInRegion; pointIndex++)
		{
			// Evaluate kernel membership functions
			// TODO: this could be optimized. if ALL membership functions are non-parametric kernel membership functions and membership (tnorm) only need to be calculated once
			Eigen::VectorXd kernelMembership(grid.inputDimensionCount);

			for (Eigen::DenseIndex xiIndex = 0; xiIndex < grid.inputDimensionCount; xiIndex++)
				kernelMembership(xiIndex) = membershipEvaluate(pointIndex, xi(xiIndex), xiIndex, xoIndex);

			// Evaluate tnorm
			membership(xoIndex, pointIndex) = tnorm(kernelMembership);

			// Store point output coordinates for posterior vector multiplication
			Eigen::DenseIndex pointGridIndex = region.pointsIndex[pointIndex];
			xoRegion(xoIndex, pointIndex) = grid.dataXo(pointGridIndex, xoIndex);
		}

		// Evaluate deffuzification
		//xo(xoIndex) = membership.row(xoIndex) * xoRegion.row(xoIndex).transpose();
		xo(xoIndex) = deffuzification(membership, xoRegion, xoIndex);
	}

	// NOTE: relies on return value optimization (RVO)
	return xo;
}

// product tnorm
double Interpolant::tnorm(const Eigen::VectorXd& kernelMembership)
{
	return kernelMembership.prod();
}

// first moment of area deffuzification
double Interpolant::deffuzification(const Eigen::MatrixXd &membership, const Eigen::MatrixXd &xoRegion, Eigen::DenseIndex xoIndex)
{
	return membership.row(xoIndex) * xoRegion.row(xoIndex).transpose();
}

double Interpolant::membershipEvaluate(Eigen::DenseIndex vertexPointIndex, double x, Eigen::DenseIndex inputIndex, Eigen::DenseIndex outputIndex)
{
	bool pxi = grid.hypercubeXi(vertexPointIndex, inputIndex);
	KFTYPE kernelFunction = kernelFunctionOptions[inputIndex];

	double membership = 0;

	switch (kernelFunction)
	{
		case KF_NEARESTNEIGHBOR: membership = kernelNearestNeighbor(x); break;
		case KF_LINEAR: membership = kernelLinear(x); break;
		case KF_CUBIC: membership = kernelCubic(x, cubicData[vertexPointIndex][pxi].f0(inputIndex, outputIndex), cubicData[vertexPointIndex][pxi].f1(inputIndex, outputIndex), cubicData[vertexPointIndex][pxi].df0(inputIndex, outputIndex), cubicData[vertexPointIndex][pxi].df1(inputIndex, outputIndex)); break; // TODO: test cubic support!
		case KF_LANCZOS: membership = kernelLanczos(x); break;
		case KF_SPLINE: membership = kernelSpline(x); break;
	}

	// if the vertex is situated on the inverse side of the dimension, invert the logic (logical complement)
	if (pxi == 1)
		membership = 1 - membership;

	return membership;
}


double Interpolant::kernelNearestNeighbor(double x)
{
	if (x < 0.5)
		return 1;
	else
		return 0;
}

double Interpolant::kernelLinear(double x)
{
	return 1 - x;
}

double Interpolant::kernelCubic(double x, double f0, double f1, double df0, double df1)
{
	// TODO: used to avoid numerical issues - is this really necessary ?
	// TODO: update value below to something sciency :)
	if (fabs(f0 - f1) < 1e-8)
		return 1;
	else
		return ((df0 + df1 + 2 * f0 - 2 * f1)*x*x*x + (3 * f1 - df1 - 3 * f0 - 2 * df0)*x*x + df0*x + f0 - f1) / (f0 - f1);
}

double Interpolant::kernelLanczos(double x)
{
	static const double pi = 3.14159265358979323846; // or, you know, atan(1)*4

	if(x == 0)
		return 1;
	else if(x > 0 && x < 1)
		return 1 * sin(pi*x)*sin(pi*x / 1) / (pi*pi * x*x);
	else // == 1
		return 0;
}

double Interpolant::kernelSpline(double x)
{
	/* taken somewhere from Matlab code
	* There is a whole family of "cubic" interpolation kernels.The
	* particular kernel used here is described in the article Keys,
	* "Cubic Convolution Interpolation for Digital Image Processing,"
	* IEEE Transactions on Acoustics, Speech, and Signal Processing,
	* Vol.ASSP - 29, No. 6, December 1981, p. 1155.
	*/
	return ((1.5 * x - 2.5)*x)*x + 1;
}

}
