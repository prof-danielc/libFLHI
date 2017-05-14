#ifndef GRIDDATA_H_INCLUDED
#define GRIDDATA_H_INCLUDED

#include <vector>
#include <Eigen/Dense>

// Extends Eigen namespace to include a VectorXui definition
namespace Eigen
{
typedef Eigen::Matrix<unsigned int, Eigen::Dynamic, Eigen::Dynamic> MatrixXui;
typedef Eigen::Matrix<unsigned int, Eigen::Dynamic, 1> VectorXui;
typedef Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic> MatrixXb;
}

namespace FLHI
{

// This could just as well be named Cell (as in a grid's cell) but the name region stuck...
class Region
{
public:
	// The order of point indices stored in pointsIndex is the same as the order of hypercubeShiftIndices and of hypercubeXi.
	// Thus, each point in pointsIndex is linearly mapped to a point in hypercube.
	// This is how a region in the grid is mapped to an hypercube.
	// Base point index is at position zero, second point index is "basePointIndex + logicalCoordinateToIndex([1 0...])", third point is "basePointIndex + logicalCoordinateToIndex([0 1 ...])", so on.
	std::vector<Eigen::DenseIndex> pointsIndex;

	Eigen::VectorXd xoMinimum;
	Eigen::VectorXd xoMaximum;

	Region(unsigned int pointCount, Eigen::DenseIndex outputDimensionCount)
		:
		pointsIndex(pointCount, 0),
		xoMinimum(outputDimensionCount),
		xoMaximum(outputDimensionCount)
	{
	}

	// NOTE: For reasons of efficiency Region depends on a move constructor/operator.
	// Current C++ standard defines default move/constructor if:
	/* C++ 14 (12.8)
	If the definition of a class X does not explicitly declare a move constructor, one will be implicitly declared
	as defaulted if and only if
	— X does not have a user-declared copy constructor,
	— X does not have a user-declared copy assignment operator,
	— X does not have a user-declared move assignment operator, and
	— X does not have a user-declared destructor
	*/
	// DO NOT DECLARE ANY OF THE ABOVE IN THIS CLASS!
};

class GridData
{
public:
	Eigen::MatrixXd dataXi;
	Eigen::MatrixXd dataXo;

	Eigen::DenseIndex pointCount;
	Eigen::DenseIndex inputDimensionCount;
	Eigen::DenseIndex outputDimensionCount;
	Eigen::DenseIndex pointCountInRegion;

	Eigen::VectorXd xiMinimum;
	Eigen::VectorXd xiMaximum;

	Eigen::VectorXd xiStep;
	Eigen::VectorXui xiLength;

	Eigen::MatrixXb hypercubeXi;
	std::vector<Eigen::DenseIndex> hypercubeShiftIndices;

	unsigned int regionCount;
	std::vector<Region> regions;

	void sortData(const Eigen::Ref<const Eigen::MatrixXd>& Xi, const Eigen::Ref<const Eigen::MatrixXd>& Xo, bool ascending=true);

	void countInputDimensionStep();
	void validateInputDimensionStep();

	void countInputDimensionLength();
	void validateInputDimensionLength();

	void groupPointsInRegions();

	void createHypercube();
	void createHypercubeShiftIndex();

	void defineRegionOutputBounds(Region &r);

	// TODO: unsure if these need to be Eigen::Ref or not, works faster this way since Ref's constructor is not called.
	// As a trade-off, if you pass an Eigen block() to any of these a copy will occur
	Eigen::VectorXd hypercubeCoordinateToSpatialCoordinate(const Eigen::VectorXd& pointHypercube, const Eigen::DenseIndex basePointIndex);

	Eigen::VectorXui spatialCoordinateToLogicalFloor(const Eigen::VectorXd& point);
	Eigen::VectorXui spatialCoordinateToLogical(const Eigen::VectorXd& point);
	Eigen::VectorXd spatialCoordinateToLength(const Eigen::VectorXd& point);
	Eigen::DenseIndex logicalCoordinateToIndex(const Eigen::VectorXui& point);
	Eigen::DenseIndex spatialCoordinateToPointIndex(const Eigen::VectorXd& point);
	Eigen::DenseIndex spatialCoordinateToPointIndexBounded(const Eigen::VectorXd& point);
	bool isPointAtMaximumBounds(const Eigen::VectorXd& point);
	bool isInternalPointAtMaximumBounds(Eigen::DenseIndex pointIndex);
	bool isPointBeyondBounds(const Eigen::VectorXd& point);
	Eigen::VectorXd limitPointToBounds(const Eigen::VectorXd& point);

	Eigen::DenseIndex logicalCoordinateToRegionIndex(const Eigen::VectorXui& point);
	Eigen::VectorXd spatialCoordinateToHypercubeCoordinate(const Eigen::VectorXd& point, const Eigen::VectorXui& logicalBasePoint);
	unsigned int findRegionIndex(const Eigen::VectorXd& point, Eigen::VectorXd& pointHypercube);
	const Region& findRegion(const Eigen::VectorXd& point, Eigen::VectorXd& pointHypercube);

public:
	GridData(const Eigen::Ref<const Eigen::MatrixXd>& pointVectorXi, const Eigen::Ref<const Eigen::MatrixXd>& pointVectorXo);

};

}

#endif // GRIDDATA_H_INCLUDED
