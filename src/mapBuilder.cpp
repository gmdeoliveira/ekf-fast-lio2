//=====================================================EKF-Fast-LIO2=====================================================================
//Institutions: Federal University of Minas Gerais (UFMG), Federal University of Ouro Preto (UFOP) and Instituto Tecnológico Vale (ITV)
//Description: This node subscribes to the EKF-filtered odometry and the Livox CustomMsg point cloud,
//             transforms each incoming scan into the world frame using a time-synchronised and
//             interpolated odometry pose, accumulates a global map, optionally publishes it,
//             and saves it as a PCD file on shutdown (Ctrl+C).
//
//  The code follows the same mapping workflow present in laserMapping.cpp:
//      - Signal handling with SigHandle / flg_exit (Ctrl+C triggers save)
//      - LiDAR callback buffering (livox_pcl_cbk pattern)
//      - Odometry callback buffering
//      - Main loop with sync_packages → process → pointBodyToWorld → accumulate
//      - publish_frame_world style publishing
//      - pcl_wait_save accumulation and final PCD save on exit
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

#include <deque>
#include <mutex>
#include <condition_variable>
#include <csignal>
#include <string>
#include <iostream>

using namespace std;
using namespace Eigen;

// ---------------------------------------------------------------------------
// Type aliases (consistent with the rest of the EKF-Fast-LIO2 codebase)
// ---------------------------------------------------------------------------
typedef pcl::PointXYZINormal PointType;
typedef pcl::PointCloud<PointType> PointCloudXYZI;

// ---------------------------------------------------------------------------
// Timestamped pose – stored in the odometry buffer
// ---------------------------------------------------------------------------
struct StampedPose
{
    double time;
    Quaterniond q;
    Vector3d    t;
};

// ---------------------------------------------------------------------------
// Global variables  (mirrors the style in laserMapping.cpp)
// ---------------------------------------------------------------------------
bool   flg_exit = false;

mutex  mtx_buffer;
condition_variable sig_buffer;

/*** Buffers ***/
deque<StampedPose>         odom_buffer;
deque<PointCloudXYZI::Ptr> lidar_buffer;
deque<double>              time_buffer;

double last_timestamp_lidar = 0;
double last_timestamp_odom  = 0;
int    scan_count = 0;

/*** Map accumulation (same pattern as pcl_wait_save in laserMapping.cpp) ***/
PointCloudXYZI::Ptr pcl_wait_save(new PointCloudXYZI());
PointCloudXYZI::Ptr pcl_wait_pub(new PointCloudXYZI(500000, 1));

/*** Voxel grid filters ***/
pcl::VoxelGrid<PointType> downSizeFilterScan;
pcl::VoxelGrid<PointType> downSizeFilterMap;

/*** Parameters ***/
string odom_topic, lidar_topic, map_topic, map_frame, save_path;
double map_voxel_size, scan_voxel_size, blind;
int    pcd_save_interval;
bool   publish_map_en;
bool   pcd_save_en = true;  // always true – the whole purpose of this node

// ---------------------------------------------------------------------------
// SigHandle  (same pattern as laserMapping.cpp)
// ---------------------------------------------------------------------------
void SigHandle(int sig)
{
    flg_exit = true;
    ROS_WARN("[mapBuilder] catch sig %d", sig);
    sig_buffer.notify_all();
}

// ---------------------------------------------------------------------------
// Interpolate between two stamped poses at a given time t ∈ [p0.time, p1.time]
// Uses SLERP for rotation and LERP for translation.
// ---------------------------------------------------------------------------
StampedPose interpolate(const StampedPose &p0, const StampedPose &p1, double t)
{
    double dt = p1.time - p0.time;
    double s  = (dt > 1e-9) ? (t - p0.time) / dt : 0.0;
    s = max(0.0, min(1.0, s));

    StampedPose out;
    out.time = t;
    out.q    = p0.q.slerp(s, p1.q);
    out.t    = (1.0 - s) * p0.t + s * p1.t;
    return out;
}

// ---------------------------------------------------------------------------
// Look up the pose at a given timestamp from the odometry buffer.
// Must be called with mtx_buffer already locked.
// ---------------------------------------------------------------------------
bool lookupPose(double stamp, StampedPose &result)
{
    if (odom_buffer.size() < 2)
        return false;

    // Scan time is before the earliest buffered odometry
    if (stamp <= odom_buffer.front().time)
    {
        result = odom_buffer.front();
        return true;
    }

    // Scan time is after the latest buffered odometry
    if (stamp >= odom_buffer.back().time)
    {
        result = odom_buffer.back();
        return true;
    }

    // Find the bracketing pair and interpolate
    for (size_t i = 0; i + 1 < odom_buffer.size(); i++)
    {
        if (odom_buffer[i].time <= stamp && stamp <= odom_buffer[i + 1].time)
        {
            result = interpolate(odom_buffer[i], odom_buffer[i + 1], stamp);
            return true;
        }
    }

    return false;
}

// ---------------------------------------------------------------------------
// pointBodyToWorld  (same pattern as laserMapping.cpp, but parameterised)
// ---------------------------------------------------------------------------
void pointBodyToWorld(PointType const * const pi, PointType * const po,
                      const Matrix3d &R, const Vector3d &t_w)
{
    Vector3d p_body(pi->x, pi->y, pi->z);
    Vector3d p_global(R * p_body + t_w);

    po->x = p_global(0);
    po->y = p_global(1);
    po->z = p_global(2);
    po->intensity = pi->intensity;
}

// ---------------------------------------------------------------------------
// Odometry callback  (mirrors imu_cbk buffering pattern)
// ---------------------------------------------------------------------------
void odom_cbk(const nav_msgs::Odometry::ConstPtr &msg)
{
    mtx_buffer.lock();

    double timestamp = msg->header.stamp.toSec();

    if (timestamp < last_timestamp_odom)
    {
        ROS_WARN("[mapBuilder] odom loop back, clear buffer");
        odom_buffer.clear();
    }
    last_timestamp_odom = timestamp;

    StampedPose sp;
    sp.time = timestamp;
    const auto &p = msg->pose.pose.position;
    const auto &q = msg->pose.pose.orientation;
    sp.q = Quaterniond(q.w, q.x, q.y, q.z).normalized();
    sp.t = Vector3d(p.x, p.y, p.z);

    odom_buffer.push_back(sp);

    // Keep the buffer bounded
    while (odom_buffer.size() > 2000)
        odom_buffer.pop_front();

    mtx_buffer.unlock();
    sig_buffer.notify_all();
}

// ---------------------------------------------------------------------------
// LiDAR callback  (mirrors livox_pcl_cbk in laserMapping.cpp)
// ---------------------------------------------------------------------------
void livox_pcl_cbk(const livox_ros_driver2::CustomMsg::ConstPtr &msg)
{
    mtx_buffer.lock();
    scan_count++;

    if (msg->header.stamp.toSec() < last_timestamp_lidar)
    {
        ROS_ERROR("[mapBuilder] lidar loop back, clear buffer");
        lidar_buffer.clear();
    }
    last_timestamp_lidar = msg->header.stamp.toSec();

    // Convert CustomMsg → PointCloudXYZI  (same filtering as preprocess avia_handler)
    const int plsize = msg->point_num;
    PointCloudXYZI::Ptr ptr(new PointCloudXYZI());
    ptr->reserve(plsize);

    for (int i = 1; i < plsize; i++)
    {
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
        pt.normal_x  = 0;
        pt.normal_y  = 0;
        pt.normal_z  = 0;
        pt.curvature = msg->points[i].offset_time / float(1000000); // ms, same as preprocess
        ptr->push_back(pt);
    }

    lidar_buffer.push_back(ptr);
    time_buffer.push_back(last_timestamp_lidar);

    mtx_buffer.unlock();
    sig_buffer.notify_all();
}

// ---------------------------------------------------------------------------
// sync_packages  (mirrors the pattern in laserMapping.cpp)
//   Pops one lidar scan and finds the matching odometry pose.
//   Returns true when a synchronised pair is available.
// ---------------------------------------------------------------------------
bool sync_packages(PointCloudXYZI::Ptr &scan_out, double &scan_time_out,
                   StampedPose &pose_out)
{
    if (lidar_buffer.empty() || odom_buffer.empty())
        return false;

    scan_time_out = time_buffer.front();

    // Need at least one odom message at or after the scan time
    if (odom_buffer.back().time < scan_time_out)
        return false;

    // Look up the interpolated pose at the scan timestamp
    if (!lookupPose(scan_time_out, pose_out))
        return false;

    scan_out = lidar_buffer.front();
    lidar_buffer.pop_front();
    time_buffer.pop_front();

    return true;
}

// ---------------------------------------------------------------------------
// publish_frame_world  (same pattern as laserMapping.cpp)
//   Transforms the scan to the world frame, publishes it, and accumulates
//   the points in pcl_wait_save.
// ---------------------------------------------------------------------------
void publish_frame_world(const ros::Publisher &pubLaserCloudFull,
                         const PointCloudXYZI::Ptr &scan_body,
                         double scan_time,
                         const Matrix3d &R, const Vector3d &t_w)
{
    int size = scan_body->points.size();
    PointCloudXYZI::Ptr laserCloudWorld(new PointCloudXYZI(size, 1));

    for (int i = 0; i < size; i++)
    {
        pointBodyToWorld(&scan_body->points[i], &laserCloudWorld->points[i], R, t_w);
    }

    /*** Publish the registered cloud ***/
    if (publish_map_en)
    {
        sensor_msgs::PointCloud2 laserCloudmsg;
        pcl::toROSMsg(*laserCloudWorld, laserCloudmsg);
        laserCloudmsg.header.stamp = ros::Time().fromSec(scan_time);
        laserCloudmsg.header.frame_id = map_frame;
        pubLaserCloudFull.publish(laserCloudmsg);
    }

    /**************** save map ****************/
    /* 1. make sure you have enough memories
    /* 2. noted that pcd save will influence the real-time performences **/
    if (pcd_save_en)
    {
        *pcl_wait_save += *laserCloudWorld;

        static int scan_wait_num = 0;
        scan_wait_num++;
        if (pcl_wait_save->size() > 0 && pcd_save_interval > 0 && scan_wait_num >= pcd_save_interval)
        {
            static int pcd_index = 0;
            pcd_index++;
            string all_points_dir(string(string(ROOT_DIR) + "PCD/scans_") + to_string(pcd_index) + string(".pcd"));
            pcl::PCDWriter pcd_writer;
            cout << "[mapBuilder] current scan saved to /PCD/" << all_points_dir << endl;
            pcd_writer.writeBinary(all_points_dir, *pcl_wait_save);
            pcl_wait_save->clear();
            scan_wait_num = 0;
        }
    }
}

// ---------------------------------------------------------------------------
// Main  (mirrors the structure of laserMapping.cpp main)
// ---------------------------------------------------------------------------
int main(int argc, char **argv)
{
    ros::init(argc, argv, "mapBuilder");
    ros::NodeHandle nh;

    /*** Load parameters ***/
    nh.param<string>("mapBuilder/odom_topic",       odom_topic,       "/filter_odom");
    nh.param<string>("mapBuilder/lidar_topic",      lidar_topic,      "/livox/lidar_CustomMsg");
    nh.param<string>("mapBuilder/map_topic",        map_topic,        "/map_cloud");
    nh.param<string>("mapBuilder/map_frame",        map_frame,        "chassis_init");
    nh.param<string>("mapBuilder/save_path",        save_path,        string(ROOT_DIR) + "PCD/globalMap.pcd");
    nh.param<double>("mapBuilder/map_voxel_size",   map_voxel_size,   0.2);
    nh.param<double>("mapBuilder/scan_voxel_size",  scan_voxel_size,  0.1);
    nh.param<int>   ("mapBuilder/pcd_save_interval",pcd_save_interval, -1);
    nh.param<double>("mapBuilder/blind",            blind,            0.5);
    nh.param<bool>  ("mapBuilder/publish_map_en",   publish_map_en,   true);

    cout << "[mapBuilder] Subscribing odometry : " << odom_topic  << endl;
    cout << "[mapBuilder] Subscribing lidar    : " << lidar_topic << endl;
    cout << "[mapBuilder] Map will be saved to : " << save_path   << endl;
    cout << "[mapBuilder] Map publishing       : " << (publish_map_en ? "ENABLED" : "DISABLED") << endl;

    /*** Set up voxel filters ***/
    if (scan_voxel_size > 0.0)
        downSizeFilterScan.setLeafSize(scan_voxel_size, scan_voxel_size, scan_voxel_size);
    if (map_voxel_size > 0.0)
        downSizeFilterMap.setLeafSize(map_voxel_size, map_voxel_size, map_voxel_size);

    /*** ROS subscribe initialization  (same pattern as laserMapping.cpp) ***/
    ros::Subscriber sub_lidar = nh.subscribe(lidar_topic, 200000, livox_pcl_cbk);
    ros::Subscriber sub_odom  = nh.subscribe(odom_topic,  200000, odom_cbk);

    ros::Publisher pubLaserCloudFull = nh.advertise<sensor_msgs::PointCloud2>
            (map_topic, 100000);

    /*** Ctrl+C handler  (same as laserMapping.cpp) ***/
    signal(SIGINT, SigHandle);
    ros::Rate rate(5000);
    bool status = ros::ok();

    while (status)
    {
        if (flg_exit) break;
        ros::spinOnce();

        PointCloudXYZI::Ptr scan_body(new PointCloudXYZI());
        double scan_time;
        StampedPose pose;

        mtx_buffer.lock();
        bool has_data = sync_packages(scan_body, scan_time, pose);
        mtx_buffer.unlock();

        if (has_data)
        {
            if (scan_body->empty())
                continue;

            /*** downsample the feature points in a scan  (same as laserMapping.cpp) ***/
            PointCloudXYZI::Ptr scan_filtered(new PointCloudXYZI());
            if (scan_voxel_size > 0.0)
            {
                downSizeFilterScan.setInputCloud(scan_body);
                downSizeFilterScan.filter(*scan_filtered);
            }
            else
            {
                scan_filtered = scan_body;
            }

            /*** Get the rotation and translation from the synchronised pose ***/
            Matrix3d R = pose.q.toRotationMatrix();
            Vector3d t_w = pose.t;

            /******* Publish points and accumulate map  (same as laserMapping.cpp) *******/
            publish_frame_world(pubLaserCloudFull, scan_filtered, scan_time, R, t_w);
        }

        status = ros::ok();
        rate.sleep();
    }

    /**************** save map  (same pattern as laserMapping.cpp exit) ****************/
    /* 1. make sure you have enough memories
    /* 2. pcd save will largely influence the real-time performences **/
    if (pcl_wait_save->size() > 0 && pcd_save_en)
    {
        // Final voxel down-sample before saving
        PointCloudXYZI::Ptr mapFiltered(new PointCloudXYZI());
        if (map_voxel_size > 0.0)
        {
            downSizeFilterMap.setInputCloud(pcl_wait_save);
            downSizeFilterMap.filter(*mapFiltered);
        }
        else
        {
            mapFiltered = pcl_wait_save;
        }

        pcl::PCDWriter pcd_writer;
        cout << "[mapBuilder] Saving final map to " << save_path << endl;
        pcd_writer.writeBinary(save_path, *mapFiltered);
        cout << "[mapBuilder] Map saved (" << mapFiltered->size() << " points)" << endl;
    }
    else
    {
        ROS_WARN("[mapBuilder] Global map is empty – nothing to save.");
    }

    return 0;
}