/**
* @author  Daniel Cavalcanti Jeronymo <danielc@utfpr.edu.br>
* @date September, 2016
*
* @section LICENSE
*
* This program is free software; you can redistribute it and/or
* modify it under the terms of the GNU General Public License as
* published by the Free Software Foundation; either version 2 of
* the License, or (at your option) any later version.
*
* This program is distributed in the hope that it will be useful, but
* WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
* General Public License for more details at
* https://www.gnu.org/copyleft/gpl.html
*
* @section DESCRIPTION
*
* GridData is a spatial data structure that allows for point access methods.
*
* It receives as inputs a column vector of points describing a point cloud. These points are sorted and then copied to GridData's internals. Input is validated for regular and semiregular grid data. In case of an irregular grid, an exception is thrown. Points are then packed into regions, which allows for time efficient access at a trade off for memory.
*
* Point access methods search the region vector and return the desired region if it contains the desired point.
*
* This class is a specialized grid for Fuzzy Logic Hypercube Interpolator (FLHI) and it considers data separation from some function Xo=f(Xi) between sets of input (Xi) and output (Xo).
*
*
* FUTURE REFERENCES:
http://numerical.recipes/CS395T/lectures2010/2010_19_LaplaceInterpolation.pdf
http://numerical.recipes/CS395T/lectures2008/20-MultidimInterp.pdf
http://www.alglib.net/interpolation/fastrbf.php
*/

#include <algorithm>
#include <stdexcept>
#include <functional>
#include <cmath>
#include "GridData.h"

namespace FLHI
{

// NOTE: helper function for floating point comparison
template<typename DerivedA, typename DerivedB>
bool allclose(const Eigen::DenseBase<DerivedA>& a,	const Eigen::DenseBase<DerivedB>& b, const typename DerivedA::RealScalar& rtol = Eigen::NumTraits<typename DerivedA::RealScalar>::dummy_precision(), const typename DerivedA::RealScalar& atol	= Eigen::NumTraits<typename DerivedA::RealScalar>::epsilon())
{
	return ((a.derived() - b.derived()).array().abs() <= (atol + rtol * b.derived().array().abs())).all();
}

bool ieee754equal(double a, double b, const double & rtol = Eigen::NumTraits<double>::dummy_precision(), const double & atol = Eigen::NumTraits<double>::epsilon())
{
    double aa = fabs(a); double ab = fabs(b);
	return fabs(a - b) <= (atol + rtol * fabs(aa > ab ? aa : ab));
}

bool ieee754lower(double a, double b, const double & rtol = Eigen::NumTraits<double>::dummy_precision(), const double & atol = Eigen::NumTraits<double>::epsilon())
{
    double aa = fabs(a); double ab = fabs(b);
	return fabs(a) < (atol + rtol * fabs(aa > ab ? aa : ab)) + fabs(b);
}

bool ieee754greater(double a, double b, const double & rtol = Eigen::NumTraits<double>::dummy_precision(), const double & atol = Eigen::NumTraits<double>::epsilon())
{
    double aa = fabs(a); double ab = fabs(b);
	return fabs(a) - (atol + rtol * fabs(aa > ab ? aa : ab)) > fabs(b);
}

Eigen::DenseIndex GridData::logicalCoordinateToRegionIndex(const Eigen::VectorXui& point)
{
	// index depends on dataXi being sorted left-to-right
	Eigen::DenseIndex index = point(0);

	for (Eigen::DenseIndex i = 1; i < inputDimensionCount; i++)
		index += point(i)*(xiLength(i - 1) - 1);

	return index;
}

Eigen::VectorXd GridData::spatialCoordinateToHypercubeCoordinate(const Eigen::VectorXd& point, const Eigen::VectorXui& logicalBasePoint)
{
	// NOTE: relies on return value optimization (RVO)
	return spatialCoordinateToLength(point) - logicalBasePoint.cast<double>();
}

unsigned int GridData::findRegionIndex(const Eigen::VectorXd& point, Eigen::VectorXd& pointHypercube)
{
	Eigen::VectorXui logicalBasePoint = spatialCoordinateToLogicalFloor(point);

	// If this point is at an edge, it doesn't form a region but is in one. It's base point will be one length shorter.
	for (Eigen::DenseIndex i = 0; i < inputDimensionCount; i++)
	{
		if (logicalBasePoint(i) == xiLength(i) - 1)
			logicalBasePoint(i)--;
	}

	// map the point's input coordinates to the hypercube in the region
	pointHypercube = spatialCoordinateToHypercubeCoordinate(point, logicalBasePoint);

	return logicalCoordinateToRegionIndex(logicalBasePoint);
}

// NOTE: this was tested using std::pair and it's about twice as slow. Let's wait for C++17's multiple return values and try again.
const Region& GridData::findRegion(const Eigen::VectorXd& point, Eigen::VectorXd& pointHypercube)
{
	unsigned int regionIndex = findRegionIndex(point, pointHypercube);

	return regions[regionIndex];
}

/**
  * Performs lexicographical sort on a column vector of points, where each point has inputDimensionCount columns.
  *
  * External data from Xi and Xo is sorted and copied to internal matrices dataXi and dataXo.
  *
  * This method is heavily inspired on IGL's sortrow:
  * https://github.com/libigl/libigl/blob/master/include/igl/sortrows.cpp
  *
  * @Xi [in] Column vector of points describing input data.
  * @Xo [in] Column vector of points describing output data.
  * @ascending [in] Optional boolean for sorting order.
*/
void GridData::sortData(const Eigen::Ref<const Eigen::MatrixXd>& Xi, const Eigen::Ref<const Eigen::MatrixXd>& Xo, bool ascending)
{
	Eigen::DenseIndex num_cols = Xi.cols();
	Eigen::DenseIndex num_rows = Xi.rows();

	// Create an index column vector
	Eigen::MatrixXi IX(num_rows, 1);
	for(Eigen::DenseIndex i = 0; i < num_rows; i++)
		IX(i) = i;

	// Create lambda functions and apply std::sort with a lexicographical criterium
	// Sort is applied on the index column vector with comparisons on X's columns
	// PS: std::lexicographical_compare could be applied here but its about 3 to 6 times slower
	if (ascending)
	{
		auto index_less_than = [&Xi, num_cols](Eigen::DenseIndex i, Eigen::DenseIndex j)
		{
			// This triggers a copy, not as effective as I had thought
			//const Eigen::VectorXd &row1 = Xi.row(i);
		    //const Eigen::VectorXd &row2 = Xi.row(j);
		    //return std::lexicographical_compare(row1.data(), row1.data() + num_cols, row2.data(), row2.data() + num_cols);
			//for (Eigen::DenseIndex c = 0; c<num_cols; c++) // right to left
			for (Eigen::DenseIndex c = num_cols; c-- > 0; ) // left to right
			{
				if (Xi.coeff(i, c) < Xi.coeff(j, c)) return true;
				else if (Xi.coeff(j, c) < Xi.coeff(i, c)) return false;
			}
			return false;
		};

		std::sort(IX.data(), IX.data() + IX.size(), index_less_than);
	}
	else
	{
		auto index_greater_than = [&Xi, num_cols](Eigen::DenseIndex i, Eigen::DenseIndex j)
		{
			//for (Eigen::DenseIndex c = 0; c<num_cols; c++) // right to left
			for (Eigen::DenseIndex c = num_cols; c-- > 0; ) // left to right
			{
				if (Xi.coeff(i, c) > Xi.coeff(j, c)) return true;
				else if (Xi.coeff(j, c) > Xi.coeff(i, c)) return false;
			}
			return false;
		};

		std::sort(IX.data(), IX.data() + IX.size(), index_greater_than);
	}

	// After the index vector is sorted, copy rows from X to our internal data
	for(Eigen::DenseIndex i = 0; i < num_rows; i++)
	{
		dataXi.row(i) = Xi.row(IX(i));
		dataXo.row(i) = Xo.row(IX(i));
	}
}

void GridData::countInputDimensionStep()
{
	for(Eigen::DenseIndex i = 0; i < inputDimensionCount; i++)
	{
		// TODO: check this for IEEE 754 floating-point issues?
		auto greater_than_min = [this, i](double value){return value > xiMinimum(i);};

		// this relies on dataXi being ColumnMajor (Eigen's default)
		double *greater = std::find_if(dataXi.col(i).data(), dataXi.col(i).data() + pointCount, greater_than_min);

		// in case a greater value is not found, this should be verified and throw a warning/exception later
		if(greater == dataXi.col(i).data() + pointCount) xiStep(i) = 0;
		else xiStep(i) = *greater - xiMinimum(i);
	}
}

void GridData::validateInputDimensionStep()
{
	// This relies on xiStep being ColumnMajor
	// TODO: check this for IEEE 754 floating-point issues?
	//bool foundZero = std::find(xiStep.data(), xiStep.data() + inputDimensionCount, 0) != (xiStep.data() + inputDimensionCount);
	//bool foundZero = xiStep.minCoeff() == 0; // old way
	bool foundZero = ieee754equal(xiStep.minCoeff(), 0);
	if(foundZero)
		throw std::runtime_error(std::string("GRIDDATA::VALIDATEINPUTDIMENSIONSTEP A zero dimension step has been detected in input data."));

	for (Eigen::DenseIndex i = 0; i < inputDimensionCount; i++)
	{
		double sumdiff = 0;
		for (Eigen::DenseIndex j = 1; j < pointCount; j++)
		{
			double diff = dataXi.coeff(j, i) - dataXi.coeff(j - 1, i);
			sumdiff += diff;

			// TODO: check this for IEEE 754 floating-point issues?
			// Insure step size is consistent
			//if(diff != xiStep(i) && dataXi.coeff(j-1, i) != xiMaximum(i) && diff != 0) //old way
			if (!ieee754equal(diff, xiStep(i)) && !ieee754equal(dataXi.coeff(j - 1, i), xiMaximum(i)) && !ieee754equal(diff, 0))
				throw std::runtime_error(std::string("GRIDDATA::VALIDATEINPUTDIMENSIONSTEP Step size is inconsistent at dimension."));
		}

		// TODO: check this for IEEE 754 floating-point issues?
		//if (sumdiff != xiMaximum(i) + xiMinimum(i)) // old way
		if (!ieee754equal(sumdiff, xiMaximum(i) + xiMinimum(i)))
			throw std::runtime_error(std::string("GRIDDATA::VALIDATEINPUTDIMENSIONSTEP Sum of step sizes does not match the expected value. Input data must be a regular grid."));
	}
}

void GridData::countInputDimensionLength()
{
	Eigen::VectorXd length = (xiMaximum - xiMinimum).cwiseQuotient(xiStep) + Eigen::VectorXd::Ones(inputDimensionCount);
	xiLength = length.unaryExpr(std::ptr_fun<double, double>(std::round)).cast<unsigned int>();
}

void GridData::validateInputDimensionLength()
{
	// this relies on xiLength being ColumnMajor
	// TODO: check this for IEEE 754 floating-point issues?
	//bool foundZero = std::find(xiLength.data(), xiLength.data() + inputDimensionCount, 0) != (xiLength.data() + inputDimensionCount);
	//bool foundZero = xiLength.minCoeff() == 0; // old way
	bool foundZero = ieee754equal(xiLength.minCoeff(), 0);
	if (foundZero)
		throw std::runtime_error(std::string("GRIDDATA::VALIDATEINPUTDIMENSIONLENGTH A zero dimension length has been detected in input data."));

    // Eigen might define Eigen::DenseIndex to be an integer, however, pointCount is zero or positive anyways.
	// TODO: do something about these useless casts
	if (static_cast<unsigned int>(pointCount) != xiLength.prod())
		throw std::runtime_error(std::string("GRIDDATA::VALIDATEINPUTDIMENSIONLENGTH Total number of input points does not match the expected point count from dimension lengths."));
}

Eigen::VectorXd GridData::hypercubeCoordinateToSpatialCoordinate(const Eigen::VectorXd& pointHypercube, const Eigen::DenseIndex basePointIndex)
{
	// slow, intuitive way
	//Eigen::VectorXd basePoint = dataXi.row(basePointIndex);
	//Eigen::VectorXd point = pointHypercube.cwiseProduct(xiStep) + basePoint;

	// faster
	Eigen::VectorXd point = pointHypercube.cwiseProduct(xiStep);
	for(Eigen::DenseIndex i = 0; i < inputDimensionCount; i++)
		point(i) += dataXi.coeff(basePointIndex, i);

	return point;
}

Eigen::VectorXui GridData::spatialCoordinateToLogicalFloor(const Eigen::VectorXd& point)
{
	// (point - minimum)./step
	Eigen::VectorXd ret = (point - xiMinimum).cwiseQuotient(xiStep.cast<double>());
	ret.unaryExpr(std::ptr_fun<double, double>(std::floor));
	return ret.cast<unsigned int>();
}

// Nearest logical
Eigen::VectorXui GridData::spatialCoordinateToLogical(const Eigen::VectorXd& point)
{
	// (point - minimum)./step
	Eigen::VectorXd ret = spatialCoordinateToLength(point);
	ret.unaryExpr(std::ptr_fun<double, double>(std::round)); // TODO: check if floor or round?
	return ret.cast<unsigned int>();
}

Eigen::VectorXd GridData::spatialCoordinateToLength(const Eigen::VectorXd& point)
{
	// (point - minimum)./step
	Eigen::VectorXd ret = (point - xiMinimum).cwiseQuotient(xiStep.cast<double>());
	return ret;
}

Eigen::DenseIndex GridData::logicalCoordinateToIndex(const Eigen::VectorXui& point)
{
	// index depends on dataXi being sorted left-to-right
	Eigen::DenseIndex index = point(0);

	for (Eigen::DenseIndex i = 1; i < inputDimensionCount; i++)
		index += point(i)*xiLength(i - 1);

	return index;
}

// TODO: this has nearest neighbor application
Eigen::DenseIndex GridData::spatialCoordinateToPointIndex(const Eigen::VectorXd& point)
{
	Eigen::VectorXui logicalPoint = spatialCoordinateToLogical(point);
	return logicalCoordinateToIndex(logicalPoint);
}

Eigen::DenseIndex GridData::spatialCoordinateToPointIndexBounded(const Eigen::VectorXd& point)
{
	Eigen::VectorXd boundedPoint = limitPointToBounds(point);

	return spatialCoordinateToPointIndex(boundedPoint);
}

bool GridData::isPointAtMaximumBounds(const Eigen::VectorXd& point)
{
	for (Eigen::DenseIndex i = 0; i < inputDimensionCount; i++)
	{
		// TODO: check this for IEEE 754 floating-point issues?
		if (point(i) == xiMaximum(i))
			return true;
	}

	return false;
}

bool GridData::isInternalPointAtMaximumBounds(Eigen::DenseIndex pointIndex)
{
	for (Eigen::DenseIndex i = 0; i < inputDimensionCount; i++)
	{
		// TODO: check this for IEEE 754 floating-point issues?
		if (dataXi(pointIndex, i) == xiMaximum(i))
			return true;
	}

	return false;
}

bool GridData::isPointBeyondBounds(const Eigen::VectorXd& point)
{
	for (Eigen::DenseIndex i = 0; i < inputDimensionCount; i++)
	{
		// TODO: check this for IEEE 754 floating-point issues?
		//if (point(i) < xiMinimum(i))
		if (ieee754lower(point(i), xiMinimum(i)))
			return true;
		else if (ieee754greater(point(i), xiMaximum(i)))
			return true;
	}

	return false;
}

Eigen::VectorXd GridData::limitPointToBounds(const Eigen::VectorXd& point)
{
	Eigen::VectorXd boundedPoint = point;

	for (Eigen::DenseIndex i = 0; i < inputDimensionCount; i++)
	{
		// TODO: check this for IEEE 754 floating-point issues?
		if (boundedPoint(i) < xiMinimum(i))
			boundedPoint(i) = xiMinimum(i);
		else if (boundedPoint(i) > xiMaximum(i))
			boundedPoint(i) = xiMaximum(i);
	}

	return boundedPoint;
}

void GridData::createHypercube()
{
	hypercubeXi = Eigen::MatrixXb::Zero(pointCountInRegion, inputDimensionCount);

	for (Eigen::DenseIndex i = 1; i < 4; i++)
	{
		bool hasCarry = true;

		hypercubeXi.row(i) = hypercubeXi.row(i - 1);

		for (Eigen::DenseIndex j = 0; j < 2 && hasCarry; j++)
		{
			if (hypercubeXi(i, j) == 1)
				hypercubeXi(i, j) = 0;
			else if (hypercubeXi(i, j) == 0)
			{
				hypercubeXi(i, j) = 1;
				hasCarry = false;
			}
		}
	}
}

void GridData::createHypercubeShiftIndex()
{
	// These shift indices when summed with a base point index result in the index of another point.
	hypercubeShiftIndices.resize(pointCountInRegion);

	for (Eigen::DenseIndex i = 0; i < pointCountInRegion; i++)
	{
		Eigen::VectorXui shiftVector = hypercubeXi.row(i).cast<unsigned int>();
		hypercubeShiftIndices[i] = logicalCoordinateToIndex(shiftVector);
	}
}

void GridData::defineRegionOutputBounds(Region &r)
{
	// DEPRECATED old way
	/*Eigen::MatrixXd xoMatrix(pointCountInRegion, outputDimensionCount);

	for (Eigen::DenseIndex i = 0; i < pointCountInRegion; i++)
		xoMatrix.row(i) = dataXo.row(r.pointsIndex[i]);

	r.xoMinimum = xoMatrix.colwise().minCoeff();
	r.xoMaximum = xoMatrix.colwise().maxCoeff();*/

	// This is much faster
	Eigen::DenseIndex pointIndex = r.pointsIndex[0];
	r.xoMinimum = dataXo.row(pointIndex);
	r.xoMaximum = dataXo.row(pointIndex);
	for (Eigen::DenseIndex i = 0; i < pointCountInRegion; i++)
	{
		pointIndex = r.pointsIndex[i];

		for (Eigen::DenseIndex j = 0; j < outputDimensionCount; j++)
		{
			double value = dataXo(pointIndex, j);
			if (value < r.xoMinimum(j))
				r.xoMinimum(j) = value;
			if (value > r.xoMaximum(j))
				r.xoMaximum(j) = value;
		}
	}

	// TODO: in case of cubic interpolation or some other invalid logic kernel function, minimum and maximum bound may extrapolate bounds from data.
	// This means that a minimization/maximization procedure must be performed in the region for interpolation.
	// In the meanwhile, min/max information from data is a good approximation.
}

void GridData::groupPointsInRegions()
{
	regions.reserve(regionCount);

	for(Eigen::DenseIndex i = 0; i < pointCount; i++)
	{
		// If this point is at an edge it can't form a new region
		if (isInternalPointAtMaximumBounds(i))
			continue;

		// Create region
		Region r(pointCountInRegion, outputDimensionCount);

		// Assign point indices to this region.
		// Point index "i" is the base point.
		// All other indices are logical upper variations of i.
		// Thus the second point is "i + logicalCoordinateToIndex([1 0...])", third point is "i + logicalCoordinateToIndex([0 1 ...])", so on.
		for (Eigen::DenseIndex j = 0; j < pointCountInRegion; j++)
			r.pointsIndex[j] = i + hypercubeShiftIndices[j];

		defineRegionOutputBounds(r);

		regions.push_back(std::move(r));
	}
}

GridData::GridData(const Eigen::Ref<const Eigen::MatrixXd>& pointVectorXi, const Eigen::Ref<const Eigen::MatrixXd>& pointVectorXo)
	:
	dataXi(pointVectorXi.rows(), pointVectorXi.cols()),
	dataXo(pointVectorXo.rows(), pointVectorXo.cols()),
	xiMinimum(pointVectorXi.cols()),
	xiMaximum(pointVectorXi.cols()),
	xiStep(pointVectorXi.cols()),
	xiLength(pointVectorXi.cols())
{
	if (pointVectorXi.cols() != pointVectorXo.cols())
		throw std::invalid_argument(std::string("GRIDDATA::GRIDDATA number of points must be the same for input and output matrices."));

	pointCount = pointVectorXi.rows();
	inputDimensionCount = pointVectorXi.cols();
	outputDimensionCount = pointVectorXo.cols();
	pointCountInRegion = 1 << inputDimensionCount; // TODO: limited to 32 bit or 64 bit depending on plataform

	if (pointVectorXi.rows() < pointCountInRegion)
		throw std::invalid_argument(std::string("GRIDDATA::GRIDDATA insufficient number of points to create regions. There must be at least two points per dimension."));

	sortData(pointVectorXi, pointVectorXo);

	xiMinimum = dataXi.row(0);
	xiMaximum = dataXi.row(pointCount-1);

	countInputDimensionStep();
	validateInputDimensionStep();

	countInputDimensionLength();
	validateInputDimensionLength();

	// If the code reached this far without throwing exceptions we have a valid uniform (regular or semiregular) grid as input.

	// calculateDerivatives();

	regionCount = (xiLength - Eigen::VectorXui::Ones(inputDimensionCount)).prod();

	createHypercube();
	createHypercubeShiftIndex();

	groupPointsInRegions();
}

}
