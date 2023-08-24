#pragma once
#include <Eigen/Core>
#include <pmp/SurfaceMesh.h>
#include <map>
//#include <unique_lock.hpp>
#include <mutex>
#include <queue>
#include <thread>
#include <condition_variable>
#include "D:/Orho/annoy/src/annoylib.h"
#include "D:/Orho/annoy/src/kissrandom.h"
namespace xxxlCGA
{
	template <class LineScalarType, bool NORM = false>
	class Line
	{
	public:
		/// The scalar type
		typedef LineScalarType ScalarType;

		/// The point type
		typedef Eigen::Vector3<LineScalarType> PointType;

		/// The line type
		typedef Line<LineScalarType, NORM> LineType;

	public:
		inline const PointType& Origin() const { return _ori; }
		inline PointType& Origin() { return _ori; }
		inline const PointType& Direction() const { return _dir; }
		/// sets the origin
		inline void SetOrigin(const PointType& ori)
		{
			_ori = ori;
		}
		/// sets the direction
		inline void SetDirection(const PointType& dir)
		{
			_dir = dir; if (NORM) _dir.normalize();
		}
		/// sets origin and direction.
		inline void Set(const PointType& ori, const PointType& dir)
		{
			SetOrigin(ori); SetDirection(dir);
		}
		//@}

		//@{
			 /** @name Constructors
			**/
			/// The empty constructor
		Line() {};
		/// The (origin, direction) constructor
		Line(const PointType& ori, const PointType& dir) { SetOrigin(ori); SetDirection(dir); };
		//@}

		inline ScalarType Projection(const  PointType& p) const
		{
			if (NORM) return ScalarType((p - _ori).dot(_dir));
			else      return ScalarType((p - _ori).dot(_dir) / _dir.squaredNorm());
		}

		inline PointType P(const ScalarType t) const
		{
			return _ori + _dir * t;
		}

	public:
		/// Origin
		PointType _ori;

		/// Direction (not necessarily normalized, unless so specified by NORM)
		PointType _dir;
	};

	class Plane
	{
	public:
		Plane() {}
		Plane(const Eigen::Vector3f& rhsV0, const Eigen::Vector3f& rhsN)
			: V0(rhsV0), n(rhsN)
		{}

	public:
		Eigen::Vector3f V0;
		Eigen::Vector3f n;
	};

	class OBB
	{
	public:
		OBB(const std::vector<Eigen::Vector3f>& pointCloud);
		OBB(const Eigen::MatrixX3d& V);
		void construct(const std::vector<Eigen::Vector3f>& pointCloud);

		/// @brief Check collision between two OBB, return true if collision happens. 
		bool overlap(const OBB& other) const;

		/// @brief Check whether the OBB contains a point.
		bool contain(const Eigen::Vector3f& p) const;

		/// @brief Width of the OBB.
		float width() const;

		/// @brief Height of the OBB.
		float height() const;

		/// @brief Depth of the OBB
		float depth() const;

		/// @brief Volume of the OBB
		float volume() const;

		/// @brief Size of the OBB (used in BV_Splitter to order two OBBs)
		float size() const;

		/// @brief Center of the OBB
		Eigen::Vector3f getCenter() const;
		void setCenter(const Eigen::Vector3f& center);

		void getAxis(Eigen::Vector3f& dir0, Eigen::Vector3f& dir1, Eigen::Vector3f& dir2) const;
		void setAxis(const Eigen::Vector3f& dir0, const Eigen::Vector3f& dir1, const Eigen::Vector3f& dir2);

		Eigen::Vector3f getExtent() const;
		void setExtent(const Eigen::Vector3f& extent);
	private:
		//计算平均值
		bool computeMeanValue(const std::vector<Eigen::Vector3f>& pointCloud, Eigen::Vector3f& meanValue);

		//计算协方差矩阵
		bool computeCovarianceMatrix(const std::vector<Eigen::Vector3f>& pointCloud, const Eigen::Vector3f& meanValue, Eigen::Matrix3f& covarianceMatrix);

		//更加协方差矩阵计算包围盒的主、中、次轴
		bool computeEigenVectors(const Eigen::Matrix3f& covariance_matrix, Eigen::Vector3f& major_axis, Eigen::Vector3f& middle_axis, Eigen::Vector3f& minor_axis, float& major_value, float& middle_value, float& minor_value);

		//计算轴向长度
		float computeLengthAlongAxis(const std::vector<Eigen::Vector3f>& pointCloud, const Eigen::Vector3f& axis, Plane& midPlane);

		//计算两个平面的交线
		int isTwoPlaneIntersected(const Plane& Pn1, const Plane& Pn2, Line<float>& isctLine);

		//计算线和平面的交点
		int isLineIntersectWithPlane(const Line<float>& line, const Plane& plane, Eigen::Vector3f& isctPt);

		bool obbDisjoint(const Eigen::Transform<float, 3, Eigen::Isometry>& tf, const Eigen::Vector3f& a, const Eigen::Vector3f& b) const;
		bool obbDisjoint(const Eigen::Matrix3f& B, const Eigen::Vector3f& T, const Eigen::Vector3f& a, const Eigen::Vector3f& b) const;
	public:
		/// @brief Orientation of OBB. The axes of the rotation matrix are the
		/// principle directions of the box. We assume that the first column
		/// corresponds to the axis with the longest box edge, second column
		/// corresponds to the shorter one and the third coulumn corresponds to the
		/// shortest one.
		Eigen::Matrix3f axis_;

		/// @brief Center of OBB
		Eigen::Vector3f center_;

		/// @brief Half dimensions of OBB
		Eigen::Vector3f extent_;
	};

	class OBBTree
	{
	public:
		// BVH 构造
		struct BVH_node {
			OBB* obb = nullptr;
			BVH_node* left = nullptr;
			BVH_node* right = nullptr;
			Eigen::MatrixX3d V;
			Eigen::MatrixX3i F;
			std::map<int, int> originalIndex;// 原有点的索引,key为现在点的索引value为在原模型上点的索引
		};
		OBBTree(const pmp::SurfaceMesh* mesh, int treeHeight);
		OBBTree(const Eigen::MatrixX3d& V, const Eigen::MatrixX3i& F, int treeHeight);//treeHeight为迭代次数即构建的二叉树的树高
		~OBBTree();

		BVH_node* buildOBBTree(const Eigen::MatrixX3d& V, const Eigen::MatrixX3i& F, BVH_node* root, std::map<int, int> originalIndex, int treeHeight);

		//计算射线与模型的交点
		std::vector<Eigen::Vector3f> rayIntersection(const Eigen::Vector3f& startPoint, const Eigen::Vector3f& dir);
		/// 更新
		bool dirtyTree(const Eigen::Matrix4f& mat);
	protected:
	private:
		//分割输入点集和面片集合
		void splitPoints(BVH_node* father
			, std::vector<Eigen::Vector3d>& vLeft_vector, std::vector<Eigen::Vector3d>& vRight_vector
			, std::vector<Eigen::Vector3i>& fLeft_vector, std::vector<Eigen::Vector3i>& fRight_vector
			, std::map<int, int>& originalLeftIndex, std::map<int, int>& originalRightIndex);
		// 定义一个递归函数来更新OBB树的每个节点
		void updateOBBTree(BVH_node* node, const Eigen::Matrix4f& mat);
		//计算射线与这个obbNode是否相交
		bool nodeIntersection(const Eigen::Vector3f& startPoint, const Eigen::Vector3f& dir, const OBB& node);
		//M-T算法
		bool rayTriangleIntersect(const Eigen::Vector3f& v0, const Eigen::Vector3f& v1, const Eigen::Vector3f& v2, const Eigen::Vector3f& orig
			, const Eigen::Vector3f& dir, float& tnear, float& u, float& v);
		//射线与立方体相交
		bool cubrIntersect(const Eigen::Vector3f& startPoint, const Eigen::Vector3f& dir, const OBB& box, float& t0, float& t1);
	public:
		BVH_node* root_ = nullptr;
		Annoy::AnnoyIndex<int, double, Annoy::Angular, Annoy::Kiss32Random, Annoy::AnnoyIndexSingleThreadedBuildPolicy>* treeAnn_ = nullptr;
	};

	class ThreadPool {
	public:
		ThreadPool(size_t num_threads) {
			for (size_t i = 0; i < num_threads; ++i) {
				threads_.emplace_back([this] {
					for (;;) {
						std::function<void()> task;
						{
							std::unique_lock<std::mutex> lock{ mutex_ };
							condition_.wait(lock, [this] { return stop_ || !tasks_.empty(); });
							if (stop_ && tasks_.empty()) {
								return;
							}
							task = move(tasks_.front());
							tasks_.pop();
						}
						task();
					}
					});
			}
		}

		~ThreadPool() {
			{
				std::unique_lock<std::mutex> lock{ mutex_ };
				stop_ = true;
			}
			condition_.notify_all();
			for (auto& thread : threads_) {
				thread.join();
			}
		}

		template<typename F>
		void enqueue(F&& f) {
			{
				std::unique_lock<std::mutex> lock{ mutex_ };
				tasks_.emplace(std::forward<F>(f));
			}
			condition_.notify_one();
		}

	private:
		std::vector<std::thread> threads_;
		std::queue<std::function<void()>> tasks_;
		std::mutex mutex_;
		std::condition_variable condition_;
		bool stop_ = false;
	};

	class CollisionDete
	{
	public:
		CollisionDete();
		~CollisionDete() {};
		/// OBBTree 1,2;内部不维护OBB树的有效性,调用前请更新OBBTree
		bool collisionDetection(OBBTree* _obbTree1, OBBTree* _obbTree2);
	private:
		bool isCollisionTree(xxxlCGA::OBBTree::BVH_node& p, xxxlCGA::OBBTree::BVH_node& q);
		// 计算两个向量在一条直线上的投影，返回投影的最小值和最大值
		void project(const Eigen::Vector3d& u, const Eigen::Vector3d& v, double& min, double& max);
		// 判断两个投影是否重叠，返回布尔值
		bool overlapPro(double min1, double max1, double min2, double max2);
		//判断两个三角形是否相交，使用分离轴理论
		bool intersect(std::vector<Eigen::Vector3d> T1, std::vector<Eigen::Vector3d> T2);
		bool TriSegIntersection(Eigen::Vector3d P0, Eigen::Vector3d P1, Eigen::Vector3d P2, Eigen::Vector3d A, Eigen::Vector3d B, Eigen::Vector3d& P);
		bool PointInTri(Eigen::Vector3d P, Eigen::Vector3d P0, Eigen::Vector3d P1, Eigen::Vector3d P2);
		double Area2(Eigen::Vector3d A, Eigen::Vector3d B, Eigen::Vector3d C);
		int dcmp(double x);

		//计算相交的叶子节点们共同的祖先
		xxxlCGA::OBBTree::BVH_node* lowestCommonAncestor(xxxlCGA::OBBTree::BVH_node* root, std::vector<xxxlCGA::OBBTree::BVH_node*>leafs);
	public:
		std::vector<std::pair<std::vector<Eigen::Vector3d>, std::vector<Eigen::Vector3d>>> tOverlap_;
		std::vector<std::pair<OBBTree*, std::vector<std::pair<std::vector<Eigen::Vector3d>, std::vector<Eigen::Vector3d>>>>> mOverlap_;
		xxxlCGA::OBBTree::BVH_node* ancestor1_ = nullptr;
		xxxlCGA::OBBTree::BVH_node* ancestor2_ = nullptr;

		std::vector<std::pair<int, double>> overlapDepth1_;
		std::vector<std::pair<int, double>> overlapDepth2_;
	private:
		std::vector<xxxlCGA::OBBTree::BVH_node*> vLeafsCollision1_;// 碰撞的叶子节点集合
		std::vector<xxxlCGA::OBBTree::BVH_node*> vLeafsCollision2_;// 碰撞的叶子节点集合
	};
}
