#include <iostream>
#include <cstdlib>
#include <chrono>
#include <vector>
using namespace std;

#include <Eigen/Dense>
using namespace Eigen;

#include "libflhi\GridData.h"
#include "libflhi\Interpolant.h"
using namespace FLHI;

int main()
{
	vector<double> v1, v2;
	for(double v = 0; v <= 1; v += 0.5) // 0.1
		v1.push_back(v);
	for (double v = 0; v <= 4; v += 2) // 0.1
		v2.push_back(v);

	MatrixXd m(v1.size()*v2.size(), 2);

	for (unsigned int i = 0; i < v1.size(); i++)
	{
		for (unsigned int j = 0; j < v2.size(); j++)
		{
			m(i*v2.size()+j, 0) = v1[i];
			m(i*v2.size() + j, 1) = v2[j];
		}
	}

	cout << m << endl;

	GridData gd(m, m);

	/*auto start = std::chrono::system_clock::now();
	for(unsigned int i = 0; i < 10000; i++)
	{
		GridData gd(m, m);
	}
	auto end = std::chrono::system_clock::now();
	std::chrono::duration<double> diff = end - start;

	cout << diff.count() << endl;
	system("pause");
	return 0;*/

	cout << endl << "Sorted" << endl << gd.dataXi << endl << endl;

	cout << endl << "Sorted" << endl << gd.dataXi << endl;

	cout << "point count " << gd.pointCount << endl;
	cout << "input count " << gd.inputDimensionCount << endl;
	cout << "output count " << gd.outputDimensionCount << endl;

	cout << "min " << gd.xiMinimum.transpose() << endl;
	cout << "max " << gd.xiMaximum.transpose() << endl;

	cout << "step " << gd.xiStep.transpose() << endl;
	cout << "length " << gd.xiLength.transpose() << endl;

	cout << "region count " << gd.regionCount << endl;
	cout << "point count in region " << gd.pointCountInRegion << endl;

	// test FLHI constructor time
	/*auto start = std::chrono::system_clock::now();
	for(unsigned int i = 0; i < 1000; i++)
	{
		Interpolant flhi(m, m, FLHI::Interpolant::KFTYPE::KF_CUBIC);
	}
	auto end = std::chrono::system_clock::now();
	std::chrono::duration<double> diff = end - start;

	cout << diff.count() << endl;
	system("pause");
	return 0;*/

	// TEST REVERSE INTERPOLATION
	Interpolant flhi(m, m, FLHI::Interpolant::KFTYPE::KF_LINEAR);
	//Interpolant flhi(m, m, FLHI::Interpolant::KFTYPE::KF_CUBIC);

	VectorXd v(2);
	v(0) = 0.934; v(1) = 3.231;
	//v(0) = 1;	v(1) = 2;

	// Since xi=m and xo=m output should match v
	cout << "Interpolated: " << flhi.interpolate(v).transpose() << endl;
	cout << "Desired: " << v.transpose() << endl;

	flhi.interpolateInverse(v);

	// test FLHI interpolate time
	/*auto start = std::chrono::system_clock::now();
	for(unsigned int i = 0; i < 50000; i++)
	{
		flhi.interpolate(v);
	}
	auto end = std::chrono::system_clock::now();
	std::chrono::duration<double> diff = end - start;

	cout << diff.count() << endl;
	system("pause");
	return 0;*/



	system("pause");
    return 0;
}
