#include "MeshBVHTree.h"
xxxlCGA::OBB::OBB(const std::vector<Eigen::Vector3f>& pointCloud)
{
	std::vector<Eigen::Vector3f> proxyPointCloud;
	proxyPointCloud.reserve(pointCloud.size());
	for (int i = 0; i < pointCloud.size(); ++i)
	{
		const Eigen::Vector3f& pt = pointCloud[i];
		Eigen::Vector3f proxyPt(pt[0], pt[1], pt[2]);
		proxyPointCloud.push_back(proxyPt);
	}

	construct(proxyPointCloud);
}

xxxlCGA::OBB::OBB(const Eigen::MatrixX3d& V)
{
	std::vector<Eigen::Vector3f> proxyPointCloud;
	proxyPointCloud.reserve(V.rows());
	for (int i = 0; i < V.rows(); ++i)
	{
		Eigen::Vector3d tt = V.row(i);
		const Eigen::Vector3f& pt = { (float)V(i,0),(float)V(i,1),(float)V(i,2) };
		Eigen::Vector3f proxyPt(pt[0], pt[1], pt[2]);
		proxyPointCloud.push_back(proxyPt);
	}

	construct(proxyPointCloud);
}

void xxxlCGA::OBB::construct(const std::vector<Eigen::Vector3f>& pointCloud)
{
	// compute mean value
	Eigen::Vector3f meanValue;
	computeMeanValue(pointCloud, meanValue);

	// compute covariance matrix
	Eigen::Matrix3f covMatrix;
	computeCovarianceMatrix(pointCloud, meanValue, covMatrix);

	Eigen::Vector3f majorAxis, midAxis, minorAxis;
	float majorValue, midValue, minorValue;
	computeEigenVectors(covMatrix, majorAxis, midAxis, minorAxis, majorValue, midValue, minorValue);

	// compute each length along covariance matrix axis
	Plane majorPlane, midPlane, minorPlane;
	float extLen0 = computeLengthAlongAxis(pointCloud, majorAxis, majorPlane);
	float extLen1 = computeLengthAlongAxis(pointCloud, midAxis, midPlane);
	float extLen2 = computeLengthAlongAxis(pointCloud, minorAxis, minorPlane);

	// compute intersect line between two plane
	Line<float> isctLine;
	isTwoPlaneIntersected(majorPlane, midPlane, isctLine);

	// compute intersect point between plane and line
	Eigen::Vector3f isctPt;
	isLineIntersectWithPlane(isctLine, minorPlane, isctPt);
	//for (int i = 0; i < pointCloud.size(); i++)
	//{
	//	center_ += pointCloud[i];
	//}
	//center_ = center_ / pointCloud.size();
	center_ = isctPt;

	// fill member data
	axis_.col(0) = majorAxis;
	axis_.col(1) = midAxis;
	axis_.col(2) = minorAxis;

	extent_(0) = extLen0 * 0.5;
	extent_(1) = extLen1 * 0.5;
	extent_(2) = extLen2 * 0.5;
}

bool xxxlCGA::OBB::overlap(const OBB& other) const
{
	/// compute the relative transform that takes us from this->frame to
	/// other.frame

	Eigen::Vector3f t = other.center_ - center_;
	Eigen::Vector3f T(axis_.col(0).dot(t), axis_.col(1).dot(t), axis_.col(2).dot(t));
	Eigen::Matrix3f R = axis_.transpose() * other.axis_;

	return !obbDisjoint(R, T, extent_, other.extent_);
}

bool xxxlCGA::OBB::contain(const Eigen::Vector3f& p) const
{
	Eigen::Vector3f local_p = p - center_;
	float proj = local_p.dot(axis_.col(0));
	if ((proj > extent_[0]) || (proj < -extent_[0]))
		return false;

	proj = local_p.dot(axis_.col(1));
	if ((proj > extent_[1]) || (proj < -extent_[1]))
		return false;

	proj = local_p.dot(axis_.col(2));
	if ((proj > extent_[2]) || (proj < -extent_[2]))
		return false;

	return true;
}

float xxxlCGA::OBB::width() const
{
	return 2 * extent_[0];
}

float xxxlCGA::OBB::height() const
{
	return 2 * extent_[1];
}

float xxxlCGA::OBB::depth() const
{
	return 2 * extent_[2];
}

float xxxlCGA::OBB::volume() const
{
	return width() * height() * depth();
}

float xxxlCGA::OBB::size() const
{
	return extent_.squaredNorm();
}

Eigen::Vector3f xxxlCGA::OBB::getCenter() const
{
	Eigen::Vector3f proxyP(center_[0], center_[1], center_[2]);
	return proxyP;
}

void xxxlCGA::OBB::setCenter(const Eigen::Vector3f& center)
{
	for (int i = 0; i < 3; ++i)
	{
		center_[i] = center[i];
	}
}

void xxxlCGA::OBB::getAxis(Eigen::Vector3f& dir0, Eigen::Vector3f& dir1, Eigen::Vector3f& dir2) const
{
	dir0 = Eigen::Vector3f(axis_(0, 0), axis_(1, 0), axis_(2, 0));
	dir1 = Eigen::Vector3f(axis_(0, 1), axis_(1, 1), axis_(2, 1));
	dir2 = Eigen::Vector3f(axis_(0, 2), axis_(1, 2), axis_(2, 2));
}

void xxxlCGA::OBB::setAxis(const Eigen::Vector3f& dir0, const Eigen::Vector3f& dir1, const Eigen::Vector3f& dir2)
{
	axis_.col(0) = Eigen::Vector3f(dir0[0], dir0[1], dir0[2]);
	axis_.col(1) = Eigen::Vector3f(dir1[0], dir1[1], dir1[2]);
	axis_.col(2) = Eigen::Vector3f(dir2[0], dir2[1], dir2[2]);
}

Eigen::Vector3f xxxlCGA::OBB::getExtent() const
{
	Eigen::Vector3f proxyExtent(extent_(0), extent_(1), extent_(2));
	return proxyExtent;
}

void xxxlCGA::OBB::setExtent(const Eigen::Vector3f& extent)
{
	for (int i = 0; i < 3; ++i)
	{
		extent_[i] = extent[i];
	}
}

bool xxxlCGA::OBB::computeMeanValue(const std::vector<Eigen::Vector3f>& pointCloud, Eigen::Vector3f& meanValue)
{
	if (pointCloud.empty())
	{
		return false;
	}
	meanValue(0) = 0.0f;
	meanValue(1) = 0.0f;
	meanValue(2) = 0.0f;

	unsigned int number_of_points = static_cast <unsigned int> (pointCloud.size());
	for (unsigned int i_point = 0; i_point < number_of_points; i_point++)
	{
		meanValue(0) += pointCloud[i_point](0);
		meanValue(1) += pointCloud[i_point](1);
		meanValue(2) += pointCloud[i_point](2);
	}

	if (number_of_points == 0)
		number_of_points = 1;

	meanValue(0) /= number_of_points;
	meanValue(1) /= number_of_points;
	meanValue(2) /= number_of_points;

	return true;
}

bool xxxlCGA::OBB::computeCovarianceMatrix(const std::vector<Eigen::Vector3f>& pointCloud, const Eigen::Vector3f& meanValue, Eigen::Matrix3f& covarianceMatrix)
{
	if (pointCloud.empty())
	{
		return false;
	}

	covarianceMatrix.setZero();

	const auto number_of_points = pointCloud.size();
	float factor = 1.0f / static_cast <float> ((number_of_points - 1 > 0) ? (number_of_points - 1) : 1);
	Eigen::Vector3f current_point;
	for (unsigned int i_point = 0; i_point < number_of_points; i_point++)
	{
		current_point(0) = pointCloud[i_point](0) - meanValue(0);
		current_point(1) = pointCloud[i_point](1) - meanValue(1);
		current_point(2) = pointCloud[i_point](2) - meanValue(2);

		covarianceMatrix += current_point * current_point.transpose();
	}

	covarianceMatrix *= factor;

	return true;
}

bool xxxlCGA::OBB::computeEigenVectors(const Eigen::Matrix3f& covariance_matrix, Eigen::Vector3f& major_axis, Eigen::Vector3f& middle_axis, Eigen::Vector3f& minor_axis, float& major_value, float& middle_value, float& minor_value)
{
	Eigen::EigenSolver <Eigen::Matrix3f> eigen_solver;
	eigen_solver.compute(covariance_matrix);

	Eigen::EigenSolver <Eigen::Matrix3f >::EigenvectorsType eigen_vectors;
	Eigen::EigenSolver <Eigen::Matrix3f >::EigenvalueType eigen_values;
	eigen_vectors = eigen_solver.eigenvectors();
	eigen_values = eigen_solver.eigenvalues();

	unsigned int temp = 0;
	unsigned int major_index = 0;
	unsigned int middle_index = 1;
	unsigned int minor_index = 2;

	if (eigen_values.real() (major_index) < eigen_values.real() (middle_index))
	{
		temp = major_index;
		major_index = middle_index;
		middle_index = temp;
	}

	if (eigen_values.real() (major_index) < eigen_values.real() (minor_index))
	{
		temp = major_index;
		major_index = minor_index;
		minor_index = temp;
	}

	if (eigen_values.real() (middle_index) < eigen_values.real() (minor_index))
	{
		temp = minor_index;
		minor_index = middle_index;
		middle_index = temp;
	}

	major_value = eigen_values.real() (major_index);
	middle_value = eigen_values.real() (middle_index);
	minor_value = eigen_values.real() (minor_index);

	major_axis = eigen_vectors.col(major_index).real();
	middle_axis = eigen_vectors.col(middle_index).real();
	minor_axis = eigen_vectors.col(minor_index).real();

	major_axis.normalize();
	middle_axis.normalize();
	minor_axis.normalize();

	float det = major_axis.dot(middle_axis.cross(minor_axis));
	if (det <= 0.0f)
	{
		major_axis(0) = -major_axis(0);
		major_axis(1) = -major_axis(1);
		major_axis(2) = -major_axis(2);
	}

	return true;
}

float xxxlCGA::OBB::computeLengthAlongAxis(const std::vector<Eigen::Vector3f>& pointCloud, const Eigen::Vector3f& axis, Plane& midPlane)
{
	if (pointCloud.empty())
	{
		return (float)-1;
	}

	Eigen::Vector3f origP(0, 0, 0);
	Line<float, false> refLine(origP, axis);

	float minT = refLine.Projection(pointCloud[0]);
	float maxT = minT;

	int pointNum = (int)pointCloud.size();
	for (int i = 1; i < pointNum; ++i)
	{
		const Eigen::Vector3f& travP = pointCloud[i];
		float candT = refLine.Projection(travP);
		if (candT < minT)
		{
			minT = candT;
		}

		if (candT > maxT)
		{
			maxT = candT;
		}
	}

	Eigen::Vector3f minP = refLine.P(minT);
	Eigen::Vector3f maxP = refLine.P(maxT);
	Eigen::Vector3f diffVec = maxP - minP;
	float len = sqrt(diffVec(0) * diffVec(0) + diffVec(1) * diffVec(1) + diffVec(2) * diffVec(2));

	Eigen::Vector3f midP = (minP + maxP) * 0.5;
	midPlane.V0 = midP;
	midPlane.n = axis;

	return len;
}

int xxxlCGA::OBB::isTwoPlaneIntersected(const Plane& Pn1, const Plane& Pn2, Line<float>& isctLine)
{
	const float epsilon = (float)0.00000001;

	Eigen::Vector3f u = Pn1.n.cross(Pn2.n);;          // cross product
	float    ax = (u(0) >= 0 ? u(0) : -u(0));
	float    ay = (u(1) >= 0 ? u(1) : -u(1));
	float    az = (u(2) >= 0 ? u(2) : -u(2));

	// test if the two planes are parallel
	if ((ax + ay + az) < epsilon) {        // Pn1 and Pn2 are near parallel
		// test if disjoint or coincide
		Eigen::Vector3f   v = Pn2.V0 - Pn1.V0;
		if (Pn1.n.dot(v) == 0)          // Pn2.V0 lies in Pn1
			return 1;                    // Pn1 and Pn2 coincide
		else
			return 0;                    // Pn1 and Pn2 are disjoint
	}

	// Pn1 and Pn2 intersect in a line
	// first determine max abs coordinate of cross product
	int      maxc;                       // max coordinate
	if (ax > ay) {
		if (ax > az)
			maxc = 1;
		else maxc = 3;
	}
	else {
		if (ay > az)
			maxc = 2;
		else maxc = 3;
	}

	// next, to get a point on the intersect line
	// zero the max coord, and solve for the other two
	Eigen::Vector3f    iP;                // intersect point
	float    d1, d2;            // the constants in the 2 plane equations
	d1 = -Pn1.n.dot(Pn1.V0);  // note: could be pre-stored  with plane
	d2 = -Pn2.n.dot(Pn2.V0);  // ditto

	switch (maxc) {             // select max coordinate
	case 1:                     // intersect with x=0
		iP(0) = 0;
		iP(1) = (d2 * Pn1.n(2) - d1 * Pn2.n(2)) / u(0);
		iP(2) = (d1 * Pn2.n(1) - d2 * Pn1.n(1)) / u(0);
		break;
	case 2:                     // intersect with y=0
		iP(0) = (d1 * Pn2.n(2) - d2 * Pn1.n(2)) / u(1);
		iP(1) = 0;
		iP(2) = (d2 * Pn1.n(0) - d1 * Pn2.n(0)) / u(1);
		break;
	case 3:                     // intersect with z=0
		iP(0) = (d2 * Pn1.n(1) - d1 * Pn2.n(1)) / u(2);
		iP(1) = (d1 * Pn2.n(0) - d2 * Pn1.n(0)) / u(2);
		iP(2) = 0;
	}

	u.normalize();
	isctLine.Set(iP, u);
	return 2;
}

int xxxlCGA::OBB::isLineIntersectWithPlane(const Line<float>& line, const Plane& plane, Eigen::Vector3f& isctPt)
{
	const float epsilon = (float)0.00000001;

	Eigen::Vector3f P0 = line.Origin();
	Eigen::Vector3f P1 = P0 + line.Direction();

	Eigen::Vector3f u = P1 - P0;
	Eigen::Vector3f w = P0 - plane.V0;

	float D = plane.n.dot(u);
	float N = -plane.n.dot(w);

	if (abs(D) < epsilon) {           // segment is parallel to plane
		if (N == 0)                      // segment lies in plane
			return 2;
		else
			return 0;                    // no intersection
	}
	// they are not parallel
	// compute intersect param
	float sI = N / D;
	//if (sI < 0 || sI > 1)
	//	return 0;                        // no intersection

	isctPt = P0 + u * sI;                  // compute segment intersect point
	return 1;
}

bool xxxlCGA::OBB::obbDisjoint(const Eigen::Transform<float, 3, Eigen::Isometry>& tf, const Eigen::Vector3f& a, const Eigen::Vector3f& b) const
{
	float t, s;
	const float reps = 1e-6;

	Eigen::Matrix3f Bf = tf.linear().cwiseAbs();
	Bf.array() += reps;

	// if any of these tests are one-sided, then the polyhedra are disjoint

	// A1 x A2 = A0
	t = ((tf.translation()[0] < 0.0) ? -tf.translation()[0] : tf.translation()[0]);

	if (t > (a[0] + Bf.row(0).dot(b)))
		return true;

	// B1 x B2 = B0
	s = tf.linear().col(0).dot(tf.translation());
	t = ((s < 0.0) ? -s : s);

	if (t > (b[0] + Bf.col(0).dot(a)))
		return true;

	// A2 x A0 = A1
	t = ((tf.translation()[1] < 0.0) ? -tf.translation()[1] : tf.translation()[1]);

	if (t > (a[1] + Bf.row(1).dot(b)))
		return true;

	// A0 x A1 = A2
	t = ((tf.translation()[2] < 0.0) ? -tf.translation()[2] : tf.translation()[2]);

	if (t > (a[2] + Bf.row(2).dot(b)))
		return true;

	// B2 x B0 = B1
	s = tf.linear().col(1).dot(tf.translation());
	t = ((s < 0.0) ? -s : s);

	if (t > (b[1] + Bf.col(1).dot(a)))
		return true;

	// B0 x B1 = B2
	s = tf.linear().col(2).dot(tf.translation());
	t = ((s < 0.0) ? -s : s);

	if (t > (b[2] + Bf.col(2).dot(a)))
		return true;

	// A0 x B0
	s = tf.translation()[2] * tf.linear()(1, 0) - tf.translation()[1] * tf.linear()(2, 0);
	t = ((s < 0.0) ? -s : s);

	if (t > (a[1] * Bf(2, 0) + a[2] * Bf(1, 0) +
		b[1] * Bf(0, 2) + b[2] * Bf(0, 1)))
		return true;

	// A0 x B1
	s = tf.translation()[2] * tf.linear()(1, 1) - tf.translation()[1] * tf.linear()(2, 1);
	t = ((s < 0.0) ? -s : s);

	if (t > (a[1] * Bf(2, 1) + a[2] * Bf(1, 1) +
		b[0] * Bf(0, 2) + b[2] * Bf(0, 0)))
		return true;

	// A0 x B2
	s = tf.translation()[2] * tf.linear()(1, 2) - tf.translation()[1] * tf.linear()(2, 2);
	t = ((s < 0.0) ? -s : s);

	if (t > (a[1] * Bf(2, 2) + a[2] * Bf(1, 2) +
		b[0] * Bf(0, 1) + b[1] * Bf(0, 0)))
		return true;

	// A1 x B0
	s = tf.translation()[0] * tf.linear()(2, 0) - tf.translation()[2] * tf.linear()(0, 0);
	t = ((s < 0.0) ? -s : s);

	if (t > (a[0] * Bf(2, 0) + a[2] * Bf(0, 0) +
		b[1] * Bf(1, 2) + b[2] * Bf(1, 1)))
		return true;

	// A1 x B1
	s = tf.translation()[0] * tf.linear()(2, 1) - tf.translation()[2] * tf.linear()(0, 1);
	t = ((s < 0.0) ? -s : s);

	if (t > (a[0] * Bf(2, 1) + a[2] * Bf(0, 1) +
		b[0] * Bf(1, 2) + b[2] * Bf(1, 0)))
		return true;

	// A1 x B2
	s = tf.translation()[0] * tf.linear()(2, 2) - tf.translation()[2] * tf.linear()(0, 2);
	t = ((s < 0.0) ? -s : s);

	if (t > (a[0] * Bf(2, 2) + a[2] * Bf(0, 2) +
		b[0] * Bf(1, 1) + b[1] * Bf(1, 0)))
		return true;

	// A2 x B0
	s = tf.translation()[1] * tf.linear()(0, 0) - tf.translation()[0] * tf.linear()(1, 0);
	t = ((s < 0.0) ? -s : s);

	if (t > (a[0] * Bf(1, 0) + a[1] * Bf(0, 0) +
		b[1] * Bf(2, 2) + b[2] * Bf(2, 1)))
		return true;

	// A2 x B1
	s = tf.translation()[1] * tf.linear()(0, 1) - tf.translation()[0] * tf.linear()(1, 1);
	t = ((s < 0.0) ? -s : s);

	if (t > (a[0] * Bf(1, 1) + a[1] * Bf(0, 1) +
		b[0] * Bf(2, 2) + b[2] * Bf(2, 0)))
		return true;

	// A2 x B2
	s = tf.translation()[1] * tf.linear()(0, 2) - tf.translation()[0] * tf.linear()(1, 2);
	t = ((s < 0.0) ? -s : s);

	if (t > (a[0] * Bf(1, 2) + a[1] * Bf(0, 2) +
		b[0] * Bf(2, 1) + b[1] * Bf(2, 0)))
		return true;

	return false;
}

bool xxxlCGA::OBB::obbDisjoint(const Eigen::Matrix3f& B, const Eigen::Vector3f& T, const Eigen::Vector3f& a, const Eigen::Vector3f& b) const
{
	float t, s;
	const float reps = 1e-6;

	Eigen::Matrix3f Bf = B.cwiseAbs();
	Bf.array() += reps;

	// if any of these tests are one-sided, then the polyhedra are disjoint

	// A1 x A2 = A0
	t = ((T[0] < 0.0) ? -T[0] : T[0]);

	if (t > (a[0] + Bf.row(0).dot(b)))
		return true;

	// B1 x B2 = B0
	s = B.col(0).dot(T);
	t = ((s < 0.0) ? -s : s);

	if (t > (b[0] + Bf.col(0).dot(a)))
		return true;

	// A2 x A0 = A1
	t = ((T[1] < 0.0) ? -T[1] : T[1]);

	if (t > (a[1] + Bf.row(1).dot(b)))
		return true;

	// A0 x A1 = A2
	t = ((T[2] < 0.0) ? -T[2] : T[2]);

	if (t > (a[2] + Bf.row(2).dot(b)))
		return true;

	// B2 x B0 = B1
	s = B.col(1).dot(T);
	t = ((s < 0.0) ? -s : s);

	if (t > (b[1] + Bf.col(1).dot(a)))
		return true;

	// B0 x B1 = B2
	s = B.col(2).dot(T);
	t = ((s < 0.0) ? -s : s);

	if (t > (b[2] + Bf.col(2).dot(a)))
		return true;

	// A0 x B0
	s = T[2] * B(1, 0) - T[1] * B(2, 0);
	t = ((s < 0.0) ? -s : s);

	if (t > (a[1] * Bf(2, 0) + a[2] * Bf(1, 0) +
		b[1] * Bf(0, 2) + b[2] * Bf(0, 1)))
		return true;

	// A0 x B1
	s = T[2] * B(1, 1) - T[1] * B(2, 1);
	t = ((s < 0.0) ? -s : s);

	if (t > (a[1] * Bf(2, 1) + a[2] * Bf(1, 1) +
		b[0] * Bf(0, 2) + b[2] * Bf(0, 0)))
		return true;

	// A0 x B2
	s = T[2] * B(1, 2) - T[1] * B(2, 2);
	t = ((s < 0.0) ? -s : s);

	if (t > (a[1] * Bf(2, 2) + a[2] * Bf(1, 2) +
		b[0] * Bf(0, 1) + b[1] * Bf(0, 0)))
		return true;

	// A1 x B0
	s = T[0] * B(2, 0) - T[2] * B(0, 0);
	t = ((s < 0.0) ? -s : s);

	if (t > (a[0] * Bf(2, 0) + a[2] * Bf(0, 0) +
		b[1] * Bf(1, 2) + b[2] * Bf(1, 1)))
		return true;

	// A1 x B1
	s = T[0] * B(2, 1) - T[2] * B(0, 1);
	t = ((s < 0.0) ? -s : s);

	if (t > (a[0] * Bf(2, 1) + a[2] * Bf(0, 1) +
		b[0] * Bf(1, 2) + b[2] * Bf(1, 0)))
		return true;

	// A1 x B2
	s = T[0] * B(2, 2) - T[2] * B(0, 2);
	t = ((s < 0.0) ? -s : s);

	if (t > (a[0] * Bf(2, 2) + a[2] * Bf(0, 2) +
		b[0] * Bf(1, 1) + b[1] * Bf(1, 0)))
		return true;

	// A2 x B0
	s = T[1] * B(0, 0) - T[0] * B(1, 0);
	t = ((s < 0.0) ? -s : s);

	if (t > (a[0] * Bf(1, 0) + a[1] * Bf(0, 0) +
		b[1] * Bf(2, 2) + b[2] * Bf(2, 1)))
		return true;

	// A2 x B1
	s = T[1] * B(0, 1) - T[0] * B(1, 1);
	t = ((s < 0.0) ? -s : s);

	if (t > (a[0] * Bf(1, 1) + a[1] * Bf(0, 1) +
		b[0] * Bf(2, 2) + b[2] * Bf(2, 0)))
		return true;

	// A2 x B2
	s = T[1] * B(0, 2) - T[0] * B(1, 2);
	t = ((s < 0.0) ? -s : s);

	if (t > (a[0] * Bf(1, 2) + a[1] * Bf(0, 2) +
		b[0] * Bf(2, 1) + b[1] * Bf(2, 0)))
		return true;

	return false;

}

xxxlCGA::OBBTree::OBBTree(const pmp::SurfaceMesh* mesh, int treeHeight)
{

}

xxxlCGA::OBBTree::OBBTree(const Eigen::MatrixX3d& V, const Eigen::MatrixX3i& F, int treeHeight)
{
	treeAnn_ = new Annoy::AnnoyIndex<int, double, Annoy::Angular, Annoy::Kiss32Random, Annoy::AnnoyIndexSingleThreadedBuildPolicy>(3);
	std::map<int, int> orgMap;
	for (int i = 0; i < V.rows(); ++i)
	{
		orgMap[i] = i;
		double tV[3] = { V.row(i).x(), V.row(i).y(), V.row(i).z() };
		treeAnn_->add_item(i, tV);
	}
	// 构建索引，构建ann树
	treeAnn_->build(10, 16);
	//// 将索引保存到磁盘
	//t.save("test.ann");
	//// 加载索引
	//t.load("test.ann");
	root_ = buildOBBTree(V, F, root_, orgMap, treeHeight);
}

xxxlCGA::OBBTree::~OBBTree()
{

}

xxxlCGA::OBBTree::BVH_node* xxxlCGA::OBBTree::buildOBBTree(const Eigen::MatrixX3d& V, const Eigen::MatrixX3i& F, BVH_node* root, std::map<int, int> originalIndex, int treeHeight)
{
	// 如果高度为0，返回空指针
	if (treeHeight == 0 || F.rows() == 0) return nullptr;
	// Base case for recursion
	if (root == nullptr)
	{
		BVH_node* tempNode = new BVH_node();
		OBB* obb = new OBB(V);
		tempNode->obb = obb;
		tempNode->V = V;
		tempNode->F = F;
		tempNode->originalIndex = originalIndex;
		tempNode->left = tempNode->right = nullptr;
		root = tempNode;
		if (F.rows() < 3)
		{
			return root;
		}

		std::vector<Eigen::Vector3d> vLeft_vector; // your vector of 3D vectors
		std::vector<Eigen::Vector3d> vRight_vector; // your vector of 3D vectors
		std::vector<Eigen::Vector3i> fLeft_vector; // your vector of 3D vectors
		std::vector<Eigen::Vector3i> fRight_vector; // your vector of 3D vectors
		std::map<int, int> originalLeftIndex;
		std::map<int, int> originalRightIndex;
		splitPoints(tempNode, vLeft_vector, vRight_vector, fLeft_vector, fRight_vector, originalLeftIndex, originalRightIndex);

		Eigen::MatrixXd leftV(vLeft_vector.size(), 3);
		for (int n = 0; n < vLeft_vector.size(); ++n) 
		{
			leftV.row(n) = vLeft_vector[n];
		}

		Eigen::MatrixXi leftF(fLeft_vector.size(), 3);
		for (int n = 0; n < fLeft_vector.size(); ++n)
		{
			leftF.row(n) = fLeft_vector[n];
		}

		Eigen::MatrixXd rightV(vRight_vector.size(), 3);
		for (int n = 0; n < vRight_vector.size(); ++n)
		{
			rightV.row(n) = vRight_vector[n];
		}

		Eigen::MatrixXi rightF(fRight_vector.size(), 3);
		for (int n = 0; n < fRight_vector.size(); ++n)
		{
			rightF.row(n) = fRight_vector[n];
		}

		// insert left child
		root->left = buildOBBTree(leftV, leftF, root->left, originalLeftIndex, treeHeight - 1);

		// insert right child
		root->right = buildOBBTree(rightV, rightF, root->right, originalRightIndex, treeHeight - 1);
	}
	return root;
}

std::vector<Eigen::Vector3f> xxxlCGA::OBBTree::rayIntersection(const Eigen::Vector3f& startPoint, const Eigen::Vector3f& dir)
{
	std::queue<BVH_node*> tNodeQue;
	std::vector<Eigen::Vector3f> retVec;
	bool isInter = nodeIntersection(startPoint, dir, *this->root_->obb);
	if (isInter)
	{
		tNodeQue.push(this->root_);
	}
	while (!tNodeQue.empty())
	{
		auto node = tNodeQue.front();
		if (node->left == nullptr && node->right == nullptr)
		{
			for (int i = 0; i < node->F.rows(); ++i)
			{
				Eigen::Vector3f v0 = node->V.row(node->F(i, 0)).cast<float>();
				Eigen::Vector3f v1 = node->V.row(node->F(i, 1)).cast<float>();
				Eigen::Vector3f v2 = node->V.row(node->F(i, 2)).cast<float>();
				float t = 0;
				float b1 = 0;
				float b2 = 0;
				if (rayTriangleIntersect(v0, v1, v2, startPoint, dir, t, b1, b2)) retVec.push_back(startPoint + t * dir);
			}
			tNodeQue.pop();
			continue;
		}
		if (node->left&& nodeIntersection(startPoint, dir, *node->left->obb))
		{
			tNodeQue.push(node->left);
		}
		if (node->right && nodeIntersection(startPoint, dir, *node->right->obb))
		{
			tNodeQue.push(node->right);
		}
		tNodeQue.pop();
	}
	return retVec;
}

void xxxlCGA::OBBTree::splitPoints(BVH_node* father
	, std::vector<Eigen::Vector3d>& vLeft_vector,std::vector<Eigen::Vector3d>& vRight_vector
	,std::vector<Eigen::Vector3i>& fLeft_vector,std::vector<Eigen::Vector3i>& fRight_vector
	, std::map<int, int>& originalLeftIndex, std::map<int, int>& originalRightIndex)
{
	std::vector<std::pair<float, Eigen::Vector3i>> sortF;
	std::vector<Eigen::Vector3d> tvLeft_vector;
	std::vector<Eigen::Vector3d> tvRight_vector;
	std::vector<Eigen::Vector3i> tfLeft_vector;
	std::vector<Eigen::Vector3i> tfRight_vector;

	std::map<int, int> leftMap;
	std::map<int, int> rightMap;
	for (int n = 0; n < father->V.rows(); ++n)
	{
		leftMap[n] = -1;
		rightMap[n] = -1;
	}

	double triArea = 0.0;
	for (int n = 0; n < father->F.rows(); ++n)
	{
		Eigen::Vector3i tFInx = father->F.row(n);
		int tempV0Inx = tFInx.x();
		int tempV1Inx = tFInx.y();
		int tempV2Inx = tFInx.z();

		double tArea = 0.5 * ((father->V.row(tempV1Inx) - father->V.row(tempV0Inx)).cross((father->V.row(tempV2Inx) - father->V.row(tempV0Inx)))).norm();
		triArea += tArea;
		Eigen::Vector3d fCent = father->V.row(tempV0Inx) + father->V.row(tempV1Inx) + father->V.row(tempV2Inx);
		fCent = fCent / 3.0;
		std::pair<float, Eigen::Vector3i> tPair(fCent.x(), tFInx);
		sortF.push_back(tPair);
	}
	std::sort(sortF.begin(), sortF.end(), [](std::pair<float, Eigen::Vector3i> a, std::pair<float, Eigen::Vector3i> b)
		{
			return a.first > b.first;
		});
	double leftArea = 0.0;
	for (int n = 0; n < sortF.size(); ++n)
	{
		Eigen::Vector3i tFInx = sortF[n].second;
		int tempV0Inx = tFInx.x();
		int tempV1Inx = tFInx.y();
		int tempV2Inx = tFInx.z();

		double tArea = 0.5 * ((father->V.row(tempV1Inx) - father->V.row(tempV0Inx)).cross((father->V.row(tempV2Inx) - father->V.row(tempV0Inx)))).norm();
		leftArea += tArea;
		if (leftArea < (triArea * 0.5))
		{
			leftMap[tempV0Inx] = 1;
			leftMap[tempV1Inx] = 1;
			leftMap[tempV2Inx] = 1;
			tfLeft_vector.push_back({ tempV0Inx ,tempV1Inx ,tempV2Inx });
		}
		else
		{
			rightMap[tempV0Inx] = 1;
			rightMap[tempV1Inx] = 1;
			rightMap[tempV2Inx] = 1;
			tfRight_vector.push_back({ tempV0Inx ,tempV1Inx ,tempV2Inx });
		}
	}
	for (auto& it : leftMap)
	{
		if (it.second == 1)
		{
			it.second = tvLeft_vector.size();
			originalLeftIndex[it.second] = father->originalIndex[it.first];
			tvLeft_vector.push_back(father->V.row(it.first));
		}
	}
	for (int n = 0; n < tfLeft_vector.size(); ++n)
	{
		tfLeft_vector[n].x() = leftMap[tfLeft_vector[n].x()];
		tfLeft_vector[n].y() = leftMap[tfLeft_vector[n].y()];
		tfLeft_vector[n].z() = leftMap[tfLeft_vector[n].z()];
	}
	for (auto& it : rightMap)
	{
		if (it.second == 1)
		{
			it.second = tvRight_vector.size();
			originalRightIndex[it.second] = father->originalIndex[it.first];
			tvRight_vector.push_back(father->V.row(it.first));
		}
	}
	for (int n = 0; n < tfRight_vector.size(); ++n)
	{
		tfRight_vector[n].x() = rightMap[tfRight_vector[n].x()];
		tfRight_vector[n].y() = rightMap[tfRight_vector[n].y()];
		tfRight_vector[n].z() = rightMap[tfRight_vector[n].z()];
	}

	vLeft_vector = tvLeft_vector;
	vRight_vector = tvRight_vector;
	fLeft_vector = tfLeft_vector;
	fRight_vector = tfRight_vector;
}

bool xxxlCGA::OBBTree::dirtyTree(const Eigen::Matrix4f& mat)
{
	updateOBBTree(root_, mat);
	return true;
}

void xxxlCGA::OBBTree::updateOBBTree(BVH_node* node, const Eigen::Matrix4f& mat)
{
	if (node == nullptr) return; // 递归终止条件

	// 更新节点的中心点和方向
	Eigen::Vector4f tCent = { node->obb->center_.x(),node->obb->center_.y(),node->obb->center_.z(),1 };
	Eigen::Matrix4f tAix = Eigen::Matrix4f();
	tAix << node->obb->axis_(0, 0), node->obb->axis_(0, 1), node->obb->axis_(0, 2), 0
		, node->obb->axis_(1, 0), node->obb->axis_(1, 1), node->obb->axis_(1, 2), 0
		, node->obb->axis_(2, 0), node->obb->axis_(2, 1), node->obb->axis_(2, 2), 0
		, 0, 0, 0, 1;
	tCent = mat * tCent;
	node->obb->center_ = { tCent.x(),tCent.y(),tCent.z() };
	tAix = mat * tAix;
	node->obb->axis_ << tAix(0, 0), tAix(0, 1), tAix(0, 2)
		, tAix(1, 0), tAix(1, 1), tAix(1, 2)
		, tAix(2, 0), tAix(2, 1), tAix(2, 2);

	// 递归更新子节点
	updateOBBTree(node->left, mat);
	updateOBBTree(node->right, mat);
}

bool xxxlCGA::OBBTree::nodeIntersection(const Eigen::Vector3f& startPoint, const Eigen::Vector3f& dir, const OBB& node)
{
	float t0 = 0.0;
	float t1 = 0.0;
	return cubrIntersect(startPoint, dir, node, t0, t1);
}

bool xxxlCGA::OBBTree::rayTriangleIntersect(const Eigen::Vector3f& v0, const Eigen::Vector3f& v1, const Eigen::Vector3f& v2, const Eigen::Vector3f& orig, const Eigen::Vector3f& dir, float& tnear, float& u, float& v)
{
	bool isIn = false;
	Eigen::Vector3f E1 = v1 - v0;
	Eigen::Vector3f E2 = v2 - v0;
	Eigen::Vector3f S = orig - v0;
	Eigen::Vector3f S1 = dir.cross(E2);
	Eigen::Vector3f S2 = S.cross(E1);
	float coeff = 1.0 / S1.dot(E1); // 共同系数
	float t = coeff * S2.dot(E2);
	float b1 = coeff * S1.dot(S);
	float b2 = coeff * S2.dot(dir);
	if (t >= 0 && b1 >= 0 && b2 >= 0 && (1 - b1 - b2) >= 0)
	{
		isIn = true;
		tnear = t;
		u = b1;
		v = b2;
	}
	return isIn;
}

bool xxxlCGA::OBBTree::cubrIntersect(const Eigen::Vector3f& startPoint, const Eigen::Vector3f& dir, const OBB& box, float& t0, float& t1)
{
	int parallelMask = 0;
	bool found = false;

	Eigen::Vector3f dirDotAxis;
	Eigen::Vector3f ocDotAxis;

	Eigen::Vector3f oc = box.getCenter() - startPoint;
	Eigen::Vector3f axisX = Eigen::Vector3f();
	Eigen::Vector3f axisY = Eigen::Vector3f();
	Eigen::Vector3f axisZ = Eigen::Vector3f();
	box.getAxis(axisX, axisY, axisZ);
	Eigen::Vector3f axis[3] = { axisX, axisY, axisZ };
	for (int i = 0; i < 3; ++i)
	{
		dirDotAxis[i] = dir.dot(axis[i]);
		ocDotAxis[i] = oc.dot(axis[i]);

		if (fabs(dirDotAxis[i]) < 0.000001)
		{
			//垂直一个方向，说明与这个方向为法线的平面平行。
			//先不处理，最后会判断是否在两个平面的区间内
			parallelMask |= 1 << i;
		}
		else
		{
			float es = (dirDotAxis[i] > 0.0f) ? box.getExtent()[i] : -box.getExtent()[i];
			float invDA = 1.0f / dirDotAxis[i]; //这个作为cos来使用，为了底下反算某轴向方向到 中心连线方向的长度
			if (!found)
			{
				// 这一步骤算出在轴向方向上，连线和平面的交点。
				// 这个平面的法线=轴
				t0 = (ocDotAxis[i] - es) * invDA;
				t1 = (ocDotAxis[i] + es) * invDA;
				found = true;
			}
			else
			{
				float s = (ocDotAxis[i] - es) * invDA;
				if (s > t0)
				{
					t0 = s;
				}
				s = (ocDotAxis[i] + es) * invDA;
				if (s < t1)
				{
					t1 = s;
				}
				if (t0 > t1)
				{
					//这里 intersect0代表就近点, intersect1代表远点。
					//t0 > t1，亦近点比远点大
					//表明了 两个t 都是负数。
					//说明了obb是在射线origin的反方向上。
					//或者是在偏移到外部擦身而过了
					return false;
				}
			}
		}
	}
	if (parallelMask)
	{
		for (int i = 0; i < 3; ++i)
		{
			if (parallelMask & (1 << i))
			{
				if (fabs(ocDotAxis[i] - t0 * dirDotAxis[i]) > box.getExtent()[i] ||
					fabs(ocDotAxis[i] - t1 * dirDotAxis[i]) > box.getExtent()[i])
				{
					return false;
				}
			}
		}
	}
	//t1 < t0已经在最上头被短路了
	if (t0 < 0)
	{
		if (t1 < 0)
		{
			return false;
		}
		t0 = t1;
	}

	return true;
}

xxxlCGA::CollisionDete::CollisionDete()
{

}

bool xxxlCGA::CollisionDete::collisionDetection(OBBTree* _obbTree1, OBBTree* _obbTree2)
{
	tOverlap_.clear();
	vLeafsCollision1_.clear();
	vLeafsCollision2_.clear();
	overlapDepth1_.clear();
	overlapDepth2_.clear();
	ancestor1_ = nullptr;
	ancestor2_ = nullptr;
	if (_obbTree1->root_ == nullptr || _obbTree2->root_ == nullptr) return false;
	auto retB = isCollisionTree(*_obbTree1->root_, *_obbTree2->root_);
	//if (retB)
	{
		ancestor1_ = lowestCommonAncestor(_obbTree1->root_, vLeafsCollision1_);
		ancestor2_ = lowestCommonAncestor(_obbTree2->root_, vLeafsCollision2_);

		if (ancestor1_)
		{
			Eigen::Vector3d refDir = (_obbTree1->root_->obb->center_ - _obbTree2->root_->obb->center_).cast<double>();
			for (int i = 0; i < ancestor1_->V.rows(); i++)
			{
				Eigen::Vector3d point = _obbTree1->root_->V.row(ancestor1_->originalIndex[i]);
				if (ancestor2_ && ancestor2_->obb->contain(Eigen::Vector3f(point.x(), point.y(), point.z())))
				{
					std::vector<int> result;
					std::vector<double> distances;
					double tV[3] = { point.x(), point.y(), point.z() };
					_obbTree2->treeAnn_->get_nns_by_vector(tV, 1, -1, &result, &distances);
					Eigen::Vector3d resultPoint = _obbTree2->root_->V.row(result[0]);
					Eigen::Vector3d tDir = (point - resultPoint);
					if (dcmp(refDir.dot(tDir)) <= 0)
					{
						overlapDepth1_.push_back(std::pair<int, double>(ancestor1_->originalIndex[i], distances[0]));
					}
				}
			}
		}
		if (ancestor2_)
		{
			Eigen::Vector3d refDir = (_obbTree2->root_->obb->center_ - _obbTree1->root_->obb->center_).cast<double>();
			for (int i = 0; i < ancestor2_->V.rows(); i++)
			{
				Eigen::Vector3d point = _obbTree2->root_->V.row(ancestor2_->originalIndex[i]);
				if (ancestor1_ && ancestor1_->obb->contain(Eigen::Vector3f(point.x(), point.y(), point.z())))
				{
					std::vector<int> result;
					std::vector<double> distances;
					double tV[3] = { point.x(), point.y(), point.z() };
					_obbTree1->treeAnn_->get_nns_by_vector(tV, 1, -1, &result, &distances);
					Eigen::Vector3d resultPoint = _obbTree1->root_->V.row(result[0]);
					Eigen::Vector3d tDir = (point - resultPoint);
					if (dcmp(refDir.dot(tDir)) <= 0)
					{
						overlapDepth2_.push_back(std::pair<int, double>(ancestor2_->originalIndex[i], distances[0]));
					}
				}
			}
		}
	}
	std::pair<OBBTree*, std::vector<std::pair<std::vector<Eigen::Vector3d>, std::vector<Eigen::Vector3d>>>> tPair = std::make_pair(_obbTree1, tOverlap_);
	mOverlap_.push_back(tPair);
	return retB;
}

bool xxxlCGA::CollisionDete::isCollisionTree(xxxlCGA::OBBTree::BVH_node& p, xxxlCGA::OBBTree::BVH_node& q)
{
	//ThreadPool pool(16);
	if (&p == nullptr || &q == nullptr)
	{
		return false;
	}
	if (!p.obb->overlap(*q.obb))
	{
		return false;
	}
	else
	{
		bool retB = false;
		if ((p.left == nullptr && p.right == nullptr) && (q.left == nullptr && q.right == nullptr))
		{
			for (int i = 0; i < p.F.rows(); ++i)
			{
				std::vector<Eigen::Vector3d> T1;
				for (int n = 0; n < 3; n++)
				{
					T1.push_back(p.V.row(p.F(i, n)));
				}
				for (int n = 0; n < q.F.rows(); ++n)
				{
					std::vector<Eigen::Vector3d> T2;
					for (int t=0;t<3;t++)
					{
						T2.push_back(q.V.row(q.F(n, t)));
					}
					if (intersect(T1, T2))
					{
						vLeafsCollision1_.push_back(&p);
						vLeafsCollision2_.push_back(&q);
						tOverlap_.push_back(std::pair(T1, T2));
						retB = true;
					}
				}
			}
			return retB;
		}
		else if ((p.left == nullptr && p.right == nullptr))
		{
			auto t1 = isCollisionTree(p, *(q.left));
			auto t2 = isCollisionTree(p, *(q.right));
			return t1 || t2;
		}
		else if ((q.left == nullptr && q.right == nullptr))
		{
			auto t1 = isCollisionTree(*(p.left), q);
			auto t2 = isCollisionTree(*(p.right), q);
			return t1 || t2;
		}
		else
		{
			auto t1 = isCollisionTree(*p.left, *q.left);
			auto t2 = isCollisionTree(*p.left, *q.right);
			auto t3 = isCollisionTree(*p.right, *q.left);
			auto t4 = isCollisionTree(*p.right, *q.right);
			return t1 || t2 || t3 || t4;
		}
	}
}

void xxxlCGA::CollisionDete::project(const Eigen::Vector3d& u, const Eigen::Vector3d& v, double& min, double& max)
{
	double p = u.dot(v); // 投影长度
	min = p;
	max = p;
}

bool xxxlCGA::CollisionDete::overlapPro(double min1, double max1, double min2, double max2)
{
	return !(min1 > max2 || min2 > max1); // 如果不重叠，返回 false，否则返回 true
}

bool xxxlCGA::CollisionDete::intersect(std::vector<Eigen::Vector3d> T1, std::vector<Eigen::Vector3d> T2)
{
	Eigen::Vector3d P;
	for (int i = 0; i < 3; i++)
	{
		if (TriSegIntersection(T1[0], T1[1], T1[2], T2[i], T2[(i + 1) % 3], P))
			return true;
		if (TriSegIntersection(T2[0], T2[1], T2[2], T1[i], T1[(i + 1) % 3], P))
			return true;
	}
	return false;
}

bool xxxlCGA::CollisionDete::TriSegIntersection(Eigen::Vector3d P0, Eigen::Vector3d P1, Eigen::Vector3d P2, Eigen::Vector3d A, Eigen::Vector3d B, Eigen::Vector3d& P)
{
	Eigen::Vector3d n = (P1 - P0).cross(P2 - P0);
	if (dcmp(n.dot(B - A) == 0)) return false;
	else
	{
		double t = n.dot(P0 - A) / n.dot(B - A);
		if (dcmp(t) < 0 || dcmp(t - 1) > 0) return false;
		P = A + (B - A) * t;
		return PointInTri(P, P0, P1, P2);
	}
}

bool xxxlCGA::CollisionDete::PointInTri(Eigen::Vector3d P, Eigen::Vector3d P0, Eigen::Vector3d P1, Eigen::Vector3d P2)
{
	double area1 = Area2(P, P0, P1);
	double area2 = Area2(P, P1, P2);
	double area3 = Area2(P, P2, P0);
	return dcmp(area1 + area2 + area3 - Area2(P0, P1, P2))==0;
}


double xxxlCGA::CollisionDete::Area2(Eigen::Vector3d A, Eigen::Vector3d B, Eigen::Vector3d C)
{
	return ((B - A).cross(C - A).norm());
}

int xxxlCGA::CollisionDete::dcmp(double x)
{
	return fabs(x) < (1e-6) ? 0 : (x > 0 ? 1 : -1);
}

xxxlCGA::OBBTree::BVH_node* xxxlCGA::CollisionDete::lowestCommonAncestor(xxxlCGA::OBBTree::BVH_node* root, std::vector<xxxlCGA::OBBTree::BVH_node*>leafs)
{
	if (root == nullptr) return nullptr;
	//我自己是叛徒之一，但是我不一定受到惩罚，因为要找的是叛徒的老板，但是你代表你的团队找到了一个叛徒
	for (auto& it : leafs)
	{
		if (root == it)
		{
			return root;
		}
	}
	//在各层级团队找叛徒
	auto first_traitor = lowestCommonAncestor(root->left, leafs);
	//在各层级团队找叛徒
	auto second_traitor = lowestCommonAncestor(root->right, leafs);

	//我们团队找到两叛徒了，game over,倒霉的就是我。接下去我会被一层一层的往上提交，最后到老板那。
	if (first_traitor != nullptr && second_traitor != nullptr) {
		return root;
	}
	//我们团队只找到一个叛徒，这个叛徒代表整个团队，就用这个叛徒来向上（回溯）提交甩锅
	//p.s. 太好了，跟我这个小manager没啥关系
	if (first_traitor != nullptr) return first_traitor;
	if (second_traitor != nullptr) return second_traitor;

	//我们是最干净的团队！
	return nullptr;
}

