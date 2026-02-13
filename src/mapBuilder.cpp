//=====================================================EKF-Fast-LIO2=====================================================================
//Institutions: Federal University of Minas Gerais (UFMG), Federal University of Ouro Preto (UFOP) and Instituto Tecnológico Vale (ITV)
//Description: This node subscribes to the EKF-filtered odometry and the Livox CustomMsg point cloud,
//             transforms each incoming scan into the world frame using the latest odometry pose,
//             accumulates a global map, optionally publishes it, and saves it as a PCD file on shutdown (Ctrl+C).
//=======================================================================================================================================

#include <ros/ros.h>
#include <nav_msgs/Odometry.h>
#include <sensor_msgs/PointCloud2.h>
#include <livox_ros_driver2/CustomMsg.h>
#include <tf/transform_datatypes.h>

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl_conversions/pcl_conversions.h>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <mutex>
#include <csignal>
#include <string>
#include <atomic>

// ---------------------------------------------------------------------------
// Type aliases (consistent with the rest of the EKF-Fast-LIO2 codebase)
// ---------------------------------------------------------------------------
typedef pcl::PointXYZINormal PointType;
typedef pcl::PointCloud<PointType> PointCloudXYZI;

// ---------------------------------------------------------------------------
// Global state
// ---------------------------------------------------------------------------
static std::atomic<bool> flg_exit(false);

// Accumulated global map
static PointCloudXYZI::Ptr globalMap(new PointCloudXYZI());

// Latest odometry pose (protected by mutex)
static std::mutex odom_mtx;
static bool odom_received = false;
static Eigen::Matrix3d R_odom = Eigen::Matrix3d::Identity();
static Eigen::Vector3d t_odom = Eigen::Vector3d::Zero();

// Parameters (loaded from the parameter server / launch file)
static std::string odom_topic;
static std::string lidar_topic;
static std::string map_topic;
static std::string map_frame;
static std::string save_path;
static double map_voxel_size;
static int    publish_interval;    // publish the map every N scans (0 = never)
static double scan_voxel_size;     // downsample each incoming scan (0 = no filter)
static double blind;               // minimum range to keep a point

// ---------------------------------------------------------------------------
// Signal handler – sets the exit flag so the main loop can save the map
// ---------------------------------------------------------------------------
void sigHandle(int sig)
{
    flg_exit.store(true);
    ROS_WARN("[mapBuilder] Caught signal %d – shutting down …", sig);
}

// ---------------------------------------------------------------------------
// Odometry callback – caches the latest pose from the EKF filter
// ---------------------------------------------------------------------------
void odomCallback(const nav_msgs::Odometry::ConstPtr &msg)
{
    std::lock_guard<std::mutex> lock(odom_mtx);

    const auto &p = msg->pose.pose.position;
    const auto &q = msg->pose.pose.orientation;

    Eigen::Quaterniond quat(q.w, q.x, q.y, q.z);
    quat.normalize();

    R_odom = quat.toRotationMatrix();
    t_odom << p.x, p.y, p.z;
    odom_received = true;
}

// ---------------------------------------------------------------------------
// CustomMsg callback – converts the Livox scan, transforms to world frame,
// and appends to the global map
// ---------------------------------------------------------------------------
void livoxCallback(const livox_ros_driver2::CustomMsg::ConstPtr &msg)
{
    // Copy the latest pose under lock
    Eigen::Matrix3d R;
    Eigen::Vector3d t;
    {
        std::lock_guard<std::mutex> lock(odom_mtx);
        if (!odom_received)
        {
            ROS_WARN_THROTTLE(2.0, "[mapBuilder] No odometry received yet – skipping scan");
            return;
        }
        R = R_odom;
        t = t_odom;
    }

    // Convert CustomMsg points into a PCL cloud in the body (LiDAR) frame
    const int plsize = msg->point_num;
    PointCloudXYZI::Ptr scan_body(new PointCloudXYZI());
    scan_body->reserve(plsize);

    for (int i = 1; i < plsize; i++)
    {
        // Skip invalid returns (same filtering as the original preprocess)
        if ((msg->points[i].tag & 0x30) != 0x10 && (msg->points[i].tag & 0x30) != 0x00)
            continue;

        double range_sq = msg->points[i].x * msg->points[i].x
                        + msg->points[i].y * msg->points[i].y
                        + msg->points[i].z * msg->points[i].z;
        if (range_sq < blind * blind)
            continue;

        PointType pt;
        pt.x = msg->points[i].x;
        pt.y = msg->points[i].y;
        pt.z = msg->points[i].z;
        pt.intensity = msg->points[i].reflectivity;
        pt.normal_x = 0;
        pt.normal_y = 0;
        pt.normal_z = 0;
        pt.curvature = msg->points[i].offset_time / 1e6f; // ms
        scan_body->push_back(pt);
    }

    if (scan_body->empty())
        return;

    // Optional per-scan voxel down-sample
    PointCloudXYZI::Ptr scan_filtered(new PointCloudXYZI());
    if (scan_voxel_size > 0.0)
    {
        pcl::VoxelGrid<PointType> vg;
        vg.setInputCloud(scan_body);
        vg.setLeafSize(scan_voxel_size, scan_voxel_size, scan_voxel_size);
        vg.filter(*scan_filtered);
    }
    else
    {
        scan_filtered = scan_body;
    }

    // Transform every point from body frame to world frame
    PointCloudXYZI::Ptr scan_world(new PointCloudXYZI());
    scan_world->resize(scan_filtered->size());

    for (size_t i = 0; i < scan_filtered->size(); i++)
    {
        const PointType &pi = scan_filtered->points[i];
        Eigen::Vector3d p_body(pi.x, pi.y, pi.z);
        Eigen::Vector3d p_world = R * p_body + t;

        PointType &po = scan_world->points[i];
        po.x = p_world(0);
        po.y = p_world(1);
        po.z = p_world(2);
        po.intensity = pi.intensity;
    }

    // Append to global map
    *globalMap += *scan_world;
}

// ---------------------------------------------------------------------------
// Save the accumulated map to a PCD file
// ---------------------------------------------------------------------------
void saveMap()
{
    if (globalMap->empty())
    {
        ROS_WARN("[mapBuilder] Global map is empty – nothing to save.");
        return;
    }

    // Final voxel down-sample before saving
    PointCloudXYZI::Ptr mapFiltered(new PointCloudXYZI());
    if (map_voxel_size > 0.0)
    {
        pcl::VoxelGrid<PointType> vg;
        vg.setInputCloud(globalMap);
        vg.setLeafSize(map_voxel_size, map_voxel_size, map_voxel_size);
        vg.filter(*mapFiltered);
    }
    else
    {
        mapFiltered = globalMap;
    }

    pcl::PCDWriter writer;
    writer.writeBinary(save_path, *mapFiltered);
    ROS_INFO("[mapBuilder] Map saved to %s  (%lu points)", save_path.c_str(), mapFiltered->size());
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main(int argc, char **argv)
{
    ros::init(argc, argv, "mapBuilder");
    ros::NodeHandle nh;
    ros::NodeHandle pnh("~");

    // ---- Parameters (with sensible defaults matching the existing config) ----
    pnh.param<std::string>("odom_topic",   odom_topic,   "/filter_odom");
    pnh.param<std::string>("lidar_topic",  lidar_topic,  "/livox/lidar_CustomMsg");
    pnh.param<std::string>("map_topic",    map_topic,    "/map_cloud");
    pnh.param<std::string>("map_frame",    map_frame,    "chassis_init");
    pnh.param<std::string>("save_path",    save_path,    std::string(ROOT_DIR) + "PCD/globalMap.pcd");
    pnh.param<double>("map_voxel_size",    map_voxel_size,  0.2);
    pnh.param<double>("scan_voxel_size",   scan_voxel_size, 0.1);
    pnh.param<int>("publish_interval",     publish_interval, 10);
    pnh.param<double>("blind",             blind,           0.5);

    ROS_INFO("[mapBuilder] Subscribing odometry : %s", odom_topic.c_str());
    ROS_INFO("[mapBuilder] Subscribing lidar    : %s", lidar_topic.c_str());
    ROS_INFO("[mapBuilder] Map will be saved to : %s", save_path.c_str());

    // ---- Subscribers ----
    ros::Subscriber sub_odom  = nh.subscribe(odom_topic,  200, odomCallback);
    ros::Subscriber sub_lidar = nh.subscribe(lidar_topic, 200, livoxCallback);

    // ---- Publisher (optional – publishes the accumulated map periodically) ----
    ros::Publisher pub_map = nh.advertise<sensor_msgs::PointCloud2>(map_topic, 1);

    // ---- Ctrl+C handler ----
    signal(SIGINT, sigHandle);

    ros::Rate rate(50);
    int scan_counter = 0;

    while (ros::ok() && !flg_exit.load())
    {
        ros::spinOnce();

        // Periodically publish the global map so it can be visualised in RViz
        if (publish_interval > 0 && !globalMap->empty())
        {
            scan_counter++;
            if (scan_counter >= publish_interval)
            {
                scan_counter = 0;

                PointCloudXYZI::Ptr mapToPublish(new PointCloudXYZI());
                if (map_voxel_size > 0.0)
                {
                    pcl::VoxelGrid<PointType> vg;
                    vg.setInputCloud(globalMap);
                    vg.setLeafSize(map_voxel_size, map_voxel_size, map_voxel_size);
                    vg.filter(*mapToPublish);
                }
                else
                {
                    mapToPublish = globalMap;
                }

                sensor_msgs::PointCloud2 mapMsg;
                pcl::toROSMsg(*mapToPublish, mapMsg);
                mapMsg.header.stamp = ros::Time::now();
                mapMsg.header.frame_id = map_frame;
                pub_map.publish(mapMsg);
            }
        }

        rate.sleep();
    }

    // ---- Save the map on exit ----
    saveMap();

    return 0;
}
