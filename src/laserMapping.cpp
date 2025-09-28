// This is an advanced implementation of the algorithm described in the
// following paper:
//   J. Zhang and S. Singh. LOAM: Lidar Odometry and Mapping in Real-time.
//     Robotics: Science and Systems Conference (RSS). Berkeley, CA, July 2014.

// Modifier: Livox               dev@livoxtech.com

// Copyright 2013, Ji Zhang, Carnegie Mellon University
// Further contributions copyright (c) 2016, Southwest Research Institute
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
// 3. Neither the name of the copyright holder nor the names of its
//    contributors may be used to endorse or promote products derived from this
//    software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
#include <omp.h>
#include <omp.h>
#include <mutex>
#include <math.h>
#include <thread>
#include <fstream>
#include <csignal>
#include <unistd.h>
#include <Python.h>
#include <so3_math.h>
#include <ros/ros.h>
#include <Eigen/Core>
#include "IMU_Processing.hpp"
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <visualization_msgs/Marker.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>
#include <geometry_msgs/Vector3.h>
#include <livox_ros_driver/CustomMsg.h>
#include "preprocess.h"
#include <ikd-Tree/ikd_Tree.h>
#include <adaptive_odom_filter/ekf_adaptive_tools.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/Image.h>
#include <nav_msgs/Odometry.h>
#include <rtabmap_msgs/ResetPose.h>

#define INIT_TIME           (0.1)
#define LASER_POINT_COV     (0.001)
#define MAXN                (720000)
#define PUBFRAME_PERIOD     (20)

class fastLioMapping
{
private:
    ros::NodeHandle nh;

    // Subscriber
    ros::Subscriber subImu;
    ros::Subscriber subWheelOdometry;
    ros::Subscriber subVisualOdometry;
    ros::Subscriber subVisualOdometryD;
    ros::Subscriber subCamLeft;
    ros::Subscriber subCamRight;
    ros::Subscriber subCamRgb;
    ros::Subscriber sub_pcl;
    ros::Subscriber sub_imu;

    // Publisher
    ros::Publisher pubFilteredOdometry;
    ros::Publisher pubLaserCloudFull;
    ros::Publisher pubLaserCloudFull_body;
    ros::Publisher pubLaserCloudEffect;
    ros::Publisher pubLaserCloudMap;
    ros::Publisher pubOdomAftMapped;
    ros::Publisher pubPath;

    // services
    ros::ServiceClient srv_client_rgbd;

    // header
    std_msgs::Header headerI;
    std_msgs::Header headerW;
    std_msgs::Header headerL;
    std_msgs::Header headerV;

    // TF 
    tf::StampedTransform filteredOdometryTrans;
    tf::TransformBroadcaster tfBroadcasterfiltered;

    // filtered odom
    nav_msgs::Odometry filteredOdometry;

    // Times
    double imuTimeLast;
    double wheelTimeLast;
    double lidarTimeLast;
    double visualTimeLast;

    double imuTimeCurrent;
    double wheelTimeCurrent;
    double lidarTimeCurrent;
    double visualTimeCurrent;

    double imu_dt;
    double wheel_dt;
    double lidar_dt;
    double visual_dt;

    // boolean
    bool imuActivated;
    bool wheelActivated;
    bool lidarActivated;
    bool visualActivated;

    // adaptive covariance - visual odometry
    double averageIntensity1;
    double averageIntensity2;
    double averageIntensity;

    // Measure
    Eigen::VectorXd imuMeasure, wheelMeasure, lidarMeasure, visualMeasure;

    // Measure Covariance
    Eigen::MatrixXd E_imu, E_wheel, E_lidar, E_visual;

    // States and covariances
    Eigen::VectorXd X;
    Eigen::MatrixXd P;

    bool firstLidarPublish = false;

    // filter constructor 
    AdaptiveOdomFilter filter;

    // auxiliar
    float cRoll, sRoll, cPitch, sPitch, cYaw, sYaw, tX, tY, tZ;

    // fast-lio
    std::shared_ptr<Preprocess> p_pre;
    std::shared_ptr<ImuProcess> p_imu;

    double timediff_lidar_wrt_imu;
    double last_timestamp_lidar, last_timestamp_imu;
    bool   timediff_set_flg = false;

    mutex mtx_buffer;

    int scan_count, publish_count;

    int effect_feat_num, frame_num;
    double deltaT, deltaR, aver_time_consu, aver_time_icp, aver_time_match, aver_time_incre, aver_time_solve, aver_time_const_H_time;
    bool flg_EKF_converged, EKF_stop_flg;
    
    deque<double> time_buffer;
    deque<PointCloudXYZI::Ptr> lidar_buffer;
    deque<sensor_msgs::Imu::ConstPtr> imu_buffer;

    double T1[MAXN], s_plot[MAXN], s_plot2[MAXN], s_plot3[MAXN], s_plot4[MAXN], s_plot5[MAXN], s_plot6[MAXN], s_plot7[MAXN], s_plot8[MAXN], s_plot9[MAXN], s_plot10[MAXN], s_plot11[MAXN];

    condition_variable sig_buffer;

    //messagens
    nav_msgs::Path path;
    nav_msgs::Odometry odomAftMapped;
    geometry_msgs::Quaternion geoQuat;
    geometry_msgs::PoseStamped msg_body_pose;

    PointCloudXYZI::Ptr featsFromMap;
    PointCloudXYZI::Ptr feats_undistort;
    PointCloudXYZI::Ptr feats_down_body;
    PointCloudXYZI::Ptr feats_down_world;
    PointCloudXYZI::Ptr normvec;
    PointCloudXYZI::Ptr laserCloudOri;
    PointCloudXYZI::Ptr corr_normvect;
    PointCloudXYZI::Ptr _featsArray;

    bool point_selected_surf[100000];   // só declara
    float res_last[100000];             // só declara

    V3F XAxisPoint_body;
    V3F XAxisPoint_world;
    V3D euler_cur;
    V3D position_last;
    V3D Lidar_T_wrt_IMU;
    M3D Lidar_R_wrt_IMU;

    double epsi[23];
    double total_residual;

    pcl::VoxelGrid<PointType> downSizeFilterSurf;
    pcl::VoxelGrid<PointType> downSizeFilterMap;

    /*** EKF inputs and output ***/
    MeasureGroup Measures;
    esekfom::esekf<state_ikfom, 12, input_ikfom> kf;
    state_ikfom state_point;
    state_ikfom state_point_map;
    vect3 pos_lid;

    string root_dir = ROOT_DIR;

    double res_mean_last;
    int    effct_feat_num, time_log_counter, iterCount, feats_down_size, laserCloudValidNum, pcd_index;
    bool   lidar_pushed, flg_first_scan, flg_exit, flg_EKF_inited;

    vector<PointVector>  Nearest_Points;
    vector<BoxPointType> cub_needrm;
    vector<vector<int>>  pointSearchInd_surf; 
    KD_TREE<PointType> ikdtree;

    /*** Time Log Variables ***/
    double kdtree_incremental_time, kdtree_search_time, kdtree_delete_time;
    double match_time, solve_time, solve_const_H_time;
    int    kdtree_size_st, kdtree_size_end, add_point_size, kdtree_delete_counter;

    double lidar_mean_scantime;
    int    scan_num;

    BoxPointType LocalMap_Points;
    bool Localmap_Initialized;

    const float MOV_THRESHOLD = 1.5f;

    ofstream fout_pre, fout_out, fout_dbg;

    int process_increments;

    PointCloudXYZI::Ptr pcl_wait_pub;
    PointCloudXYZI::Ptr pcl_wait_save;

    FILE *fp;

public:
    // Ponteiro estático para singleton
    static fastLioMapping* instance;

    // filter
    bool enableFilter;
    bool enableImu;
    bool enableWheel;
    bool enableLidar;
    bool enableVisual;

    double freq;

    double alpha_lidar;
    double alpha_visual;

    float lidarG;
    float visualG;
    float wheelG;
    float imuG;

    float gamma_vx;
    float gamma_omegaz;
    float delta_vx;
    float delta_omegaz;

    double minIntensity; 
    double maxIntensity;

    int lidar_type_func;
    int visual_type_func;
    int wheel_type_func;

    int camera_type;

    std::string filterFreq;

    // fast-lio
    bool path_en, scan_pub_en, dense_pub_en, scan_body_pub_en, time_sync_en;
    bool feature_extract_enable, runtime_pos_log, extrinsic_est_en, pcd_save_en;
    bool publish_map;

    int NUM_MAX_ITERATIONS, pcd_save_interval;
    int lidar_type, scan_line, timestamp_unit, scan_rate, point_filter_num;

    double time_diff_lidar_to_imu;
    double filter_size_corner_min, filter_size_surf_min, filter_size_map_min, fov_deg, cube_len;
    double gyr_cov, acc_cov, b_gyr_cov, b_acc_cov;
    double blind;
    double HALF_FOV_COS = 0, FOV_DEG = 0, total_distance = 0, lidar_end_time = 0, first_lidar_time = 0.0;

    float DET_RANGE;

    vector<double> extrinT;
    vector<double> extrinR;

    string map_file_path, lid_topic, imu_topic;

    // Construtor
    fastLioMapping():
        nh("~")
    {
        // ponteiro global
        instance = this;
        // Initialization
        initialization();

    }

    //--------------------------
    // auxiliar functions
    //--------------------------
    void initialization_adaptive_filter(){
        // setting the filter
        filter.enableImu = enableImu;
        filter.enableWheel = enableWheel;
        filter.enableLidar = enableLidar;
        filter.enableVisual = enableVisual;
        filter.lidar_type_func = lidar_type_func;
        filter.visual_type_func = visual_type_func;
        filter.wheel_type_func = wheel_type_func;

        filter.freq = freq;
        // there are other parameters to set, i.e., the priori state with your covariance matrix
    }

    void initialization(){
        // adaptive filter
        // times
        imuTimeLast = 0;
        lidarTimeLast = 0;
        visualTimeLast = 0;
        wheelTimeLast = 0;

        process_increments = 0;

        imuTimeCurrent = 0;
        lidarTimeCurrent = 0;
        visualTimeCurrent = 0;
        wheelTimeCurrent = 0;

        imu_dt = 0.005;
        wheel_dt = 0.05;
        lidar_dt = 0.1;
        visual_dt = 0.005;

        alpha_visual = 0.98;
        alpha_lidar = 0.98;

        // filter 
        freq = 200.0;

        // boolean
        imuActivated = false;
        lidarActivated = false;
        wheelActivated = false;
        visualActivated = false;

        enableFilter = false;
        enableImu = false;
        enableWheel = false;
        enableLidar = false;
        enableVisual = false;

        publish_map = false;

        wheelG = 0; // delete
        imuG = 0;

        // adaptive covariance - visual odometry
        averageIntensity1 = 0;
        averageIntensity2 = 0;
        averageIntensity = 0;

        // measure
        imuMeasure.resize(9);
        wheelMeasure.resize(2);
        lidarMeasure.resize(6);
        visualMeasure.resize(6);

        imuMeasure = Eigen::VectorXd::Zero(9);
        wheelMeasure = Eigen::VectorXd::Zero(2);
        lidarMeasure = Eigen::VectorXd::Zero(6);
        visualMeasure = Eigen::VectorXd::Zero(6);

        E_imu.resize(9,9);
        E_wheel.resize(2,2);
        E_lidar.resize(6,6);
        E_visual.resize(6,6);

        E_imu = Eigen::MatrixXd::Zero(9,9);
        E_lidar = Eigen::MatrixXd::Zero(6,6);
        E_visual = Eigen::MatrixXd::Zero(6,6);
        E_wheel = Eigen::MatrixXd::Zero(2,2);

        X.resize(12);
        P.resize(12,12);
        X = Eigen::VectorXd::Zero(12);
        P = Eigen::MatrixXd::Zero(12,12);

        // fast-lio2
        path_en = true;
        scan_pub_en = false;
        dense_pub_en = false;
        scan_body_pub_en = false;
        time_sync_en = false;
        feature_extract_enable = false;
        runtime_pos_log = false;
        extrinsic_est_en = true;
        pcd_save_en = false;

        NUM_MAX_ITERATIONS = 0;
        lidar_type = AVIA;
        timestamp_unit = US;
        scan_line = 16;
        scan_rate = 10;
        point_filter_num = 2;
        pcd_save_interval = -1;

        time_diff_lidar_to_imu = 0.0;
        filter_size_corner_min = 0.0;
        filter_size_surf_min = 0.0;
        filter_size_map_min = 0.0;
        fov_deg = 0.0;
        cube_len = 0.0;

        total_residual = 0.0;

        blind = 0.01;

        gyr_cov = 0.1;
        acc_cov = 0.1;
        b_gyr_cov = 0.0001;
        b_acc_cov = 0.0001;

        DET_RANGE = 300.0f;

        extrinT = std::vector<double>(3, 0.0);  // 3 elementos com valor 0.0
        extrinR = std::vector<double>(9, 0.0);  // 9 elementos com valor 0.0

        p_pre = std::make_shared<Preprocess>();
        p_imu = std::make_shared<ImuProcess>();

        timediff_lidar_wrt_imu = 0.0;
        timediff_set_flg = false;
        last_timestamp_lidar = 0;
        last_timestamp_imu = -1.0;

        scan_count = 0;
        publish_count = 0;

        effect_feat_num = 0;
        frame_num = 0;
        aver_time_consu = 0;
        aver_time_icp = 0;
        aver_time_match = 0;
        aver_time_incre = 0;
        aver_time_solve = 0;
        aver_time_const_H_time = 0;
        EKF_stop_flg = false;

        HALF_FOV_COS = 0;
        FOV_DEG = 0;
        total_distance = 0;
        lidar_end_time = 0;
        first_lidar_time = 0.0;

        std::fill_n(point_selected_surf, 100000, false);
        std::fill_n(res_last, 100000, 0.0f);
        
        std::fill(epsi, epsi + 23, 0.001);

        XAxisPoint_body = V3F(LIDAR_SP_LEN, 0.0f, 0.0f);  
        XAxisPoint_world = V3F(LIDAR_SP_LEN, 0.0f, 0.0f);
        position_last = Zero3d;
        Lidar_T_wrt_IMU = Zero3d;
        Lidar_R_wrt_IMU = Eye3d;

        featsFromMap    = PointCloudXYZI::Ptr(new PointCloudXYZI());
        feats_undistort = PointCloudXYZI::Ptr(new PointCloudXYZI());
        feats_down_body = PointCloudXYZI::Ptr(new PointCloudXYZI());
        feats_down_world= PointCloudXYZI::Ptr(new PointCloudXYZI());
        normvec         = PointCloudXYZI::Ptr(new PointCloudXYZI(100000, 1));
        laserCloudOri   = PointCloudXYZI::Ptr(new PointCloudXYZI(100000, 1));
        corr_normvect   = PointCloudXYZI::Ptr(new PointCloudXYZI(100000, 1));

        res_mean_last = 0.05;
        effct_feat_num = 0;
        time_log_counter = 0;
        iterCount = 0;
        feats_down_size = 0;
        laserCloudValidNum = 0;
        pcd_index = 0;
        flg_first_scan = true;
        flg_exit = false;

        /*** Time Log Variables ***/
        kdtree_incremental_time = 0.0;
        kdtree_search_time = 0.0;
        kdtree_delete_time = 0.0;
        match_time = 0;
        solve_time = 0;
        solve_const_H_time = 0;
        kdtree_size_st = 0;
        kdtree_size_end = 0;
        add_point_size = 0;
        kdtree_delete_counter = 0;

        lidar_mean_scantime = 0.0;
        scan_num = 0;

        Localmap_Initialized = false;

        pcl_wait_pub.reset(new PointCloudXYZI(500000, 1));
        pcl_wait_save.reset(new PointCloudXYZI());
    }

    void fastlio_initialization(){
        p_pre->blind = blind;
        p_pre->N_SCANS = scan_line;
        p_pre->time_unit = timestamp_unit;
        p_pre->SCAN_RATE = scan_rate;
        p_pre->point_filter_num = point_filter_num;
        p_pre->feature_enabled = feature_extract_enable;
        p_pre->lidar_type = lidar_type;

        // path message
        path.header.stamp    = ros::Time::now();
        path.header.frame_id ="camera_init";

        FOV_DEG = (fov_deg + 10.0) > 179.9 ? 179.9 : (fov_deg + 10.0);
        HALF_FOV_COS = cos((FOV_DEG) * 0.5 * PI_M / 180.0);

        _featsArray.reset(new PointCloudXYZI());

        memset(point_selected_surf, true, sizeof(point_selected_surf));
        memset(res_last, -1000.0f, sizeof(res_last));
        downSizeFilterSurf.setLeafSize(filter_size_surf_min, filter_size_surf_min, filter_size_surf_min);
        downSizeFilterMap.setLeafSize(filter_size_map_min, filter_size_map_min, filter_size_map_min);
        memset(point_selected_surf, true, sizeof(point_selected_surf));
        memset(res_last, -1000.0f, sizeof(res_last));

        Lidar_T_wrt_IMU<<VEC_FROM_ARRAY(extrinT);
        Lidar_R_wrt_IMU<<MAT_FROM_ARRAY(extrinR);
        p_imu->set_extrinsic(Lidar_T_wrt_IMU, Lidar_R_wrt_IMU);
        p_imu->set_gyr_cov(V3D(gyr_cov, gyr_cov, gyr_cov));
        p_imu->set_acc_cov(V3D(acc_cov, acc_cov, acc_cov));
        p_imu->set_gyr_bias_cov(V3D(b_gyr_cov, b_gyr_cov, b_gyr_cov));
        p_imu->set_acc_bias_cov(V3D(b_acc_cov, b_acc_cov, b_acc_cov));
        p_imu->lidar_type = lidar_type;
        double epsi[23] = {0.001};
        fill(epsi, epsi+23, 0.001);
        kf.init_dyn_share(get_f, df_dx, df_dw, fastLioMapping::h_share_model, NUM_MAX_ITERATIONS, epsi); // verificar aqui a funcao
        
        /*** debug record ***/
        string pos_log_dir = root_dir + "/Log/pos_log.txt";
        fp = fopen(pos_log_dir.c_str(),"w");

        fout_pre.open(DEBUG_FILE_DIR("mat_pre.txt"),ios::out);
        fout_out.open(DEBUG_FILE_DIR("mat_out.txt"),ios::out);
        fout_dbg.open(DEBUG_FILE_DIR("dbg.txt"),ios::out);
        if (fout_pre && fout_out)
            cout << "~~~~"<<ROOT_DIR<<" file opened" << endl;
        else
            cout << "~~~~"<<ROOT_DIR<<" doesn't exist" << endl;

    }

    void ros_topic_initialization(){
        // Subscriber  
        subWheelOdometry = nh.subscribe<nav_msgs::Odometry>("/odom", 5, &fastLioMapping::wheelOdometryHandler, this);        
        if (camera_type==1){
            subVisualOdometry = nh.subscribe<nav_msgs::Odometry>("/tracking_odom", 5, &fastLioMapping::visualOdometryHandler, this);
            subCamLeft = nh.subscribe<sensor_msgs::Image>("/left_camera", 5, &fastLioMapping::camLeftHandler, this);
            subCamRight = nh.subscribe<sensor_msgs::Image>("/rigth_camera", 5, &fastLioMapping::camRightHandler, this);
        }else if (camera_type==2){
            subVisualOdometryD = nh.subscribe<nav_msgs::Odometry>("/depth_odom", 5, &fastLioMapping::visualOdometryDHandler, this);
            subCamRgb = nh.subscribe<sensor_msgs::Image>("/camera_color", 5, &fastLioMapping::camRgbHandler, this);
        }
        // fastlio
        if (p_pre->lidar_type == AVIA){ 
            sub_pcl = nh.subscribe<livox_ros_driver::CustomMsg>(lid_topic, 200000, &fastLioMapping::livox_pcl_cbk, this);
        }else{
            sub_pcl = nh.subscribe<sensor_msgs::PointCloud2>(lid_topic, 200000, &fastLioMapping::standard_pcl_cbk, this);
        }
        sub_imu = nh.subscribe<sensor_msgs::Imu>(imu_topic, 200000, &fastLioMapping::imu_cbk, this); // dividido entre fastlio e adaptive filter
          
        // Publisher
        pubFilteredOdometry = nh.advertise<nav_msgs::Odometry> ("/filter_odom", 5);
        // fastlio
        pubLaserCloudFull = nh.advertise<sensor_msgs::PointCloud2>("/cloud_registered", 100000);
        pubLaserCloudFull_body = nh.advertise<sensor_msgs::PointCloud2>("/cloud_registered_body", 100000);
        pubLaserCloudEffect = nh.advertise<sensor_msgs::PointCloud2>("/cloud_effected", 100000);
        pubLaserCloudMap = nh.advertise<sensor_msgs::PointCloud2>("/Laser_map", 100000);
        pubOdomAftMapped = nh.advertise<nav_msgs::Odometry>("/Odometry", 100000);
        pubPath = nh.advertise<nav_msgs::Path>("/path", 100000);

        // Services
        srv_client_rgbd = nh.serviceClient<rtabmap_msgs::ResetPose>("/rtabmap/reset_odom_to_pose");       
    }
  
    //----------
    // callbacks
    //----------
    void wheelOdometryHandler(const nav_msgs::Odometry::ConstPtr& wheelOdometry){
        double timeL = ros::Time::now().toSec();

        // time
        if (wheelActivated){
            wheelTimeLast = wheelTimeCurrent;
            wheelTimeCurrent = wheelOdometry->header.stamp.toSec();
        }else{
            wheelTimeCurrent = wheelOdometry->header.stamp.toSec();
            wheelTimeLast = wheelTimeCurrent + 0.01;
            wheelActivated = true;
        } 

        // measure
        wheelMeasure << -1.0*wheelOdometry->twist.twist.linear.x, wheelOdometry->twist.twist.angular.z;

        // covariance
        E_wheel(0,0) = wheelG*wheelOdometry->twist.covariance[0];
        E_wheel(1,1) = 100*wheelOdometry->twist.covariance[35];

        // time
        wheel_dt = wheelTimeCurrent - wheelTimeLast;

        // header
        double timediff = ros::Time::now().toSec() - timeL + wheelTimeCurrent;
        headerW = wheelOdometry->header;
        headerW.stamp = ros::Time().fromSec(timediff);

        // correction stage aqui
        filter.correction_wheel_data(wheelMeasure, E_wheel, wheel_dt, imuMeasure(5));
    }

    void visualOdometryHandler(const nav_msgs::Odometry::ConstPtr& visualOdometry){
        if (camera_type == 1){ 
            double timeV = ros::Time::now().toSec();

            if (visualActivated){
                visualTimeLast = visualTimeCurrent;
                visualTimeCurrent = visualOdometry->header.stamp.toSec();
            }else{
                visualTimeCurrent = visualOdometry->header.stamp.toSec();
                visualTimeLast = visualTimeCurrent + 0.01;
                visualActivated = true;
            }  
            
            // roll, pitch and yaw 
            double roll, pitch, yaw;
            geometry_msgs::Quaternion orientation = visualOdometry->pose.pose.orientation;
            tf::Matrix3x3(tf::Quaternion(orientation.x, orientation.y, orientation.z, orientation.w)).getRPY(roll, pitch, yaw);

            visualMeasure.block(0,0,3,1) << visualOdometry->pose.pose.position.x, visualOdometry->pose.pose.position.y, visualOdometry->pose.pose.position.z;
            visualMeasure.block(3,0,3,1) << roll, pitch, yaw;    

            // covariance
            E_visual(0,0) = visualOdometry->pose.covariance[0];
            E_visual(0,1) = visualOdometry->pose.covariance[1];
            E_visual(0,2) = visualOdometry->pose.covariance[2];
            E_visual(1,0) = visualOdometry->pose.covariance[3];
            E_visual(1,1) = visualOdometry->pose.covariance[4];
            E_visual(1,2) = visualOdometry->pose.covariance[5];
            E_visual(2,0) = visualOdometry->pose.covariance[6];
            E_visual(2,1) = visualOdometry->pose.covariance[7];
            E_visual(2,2) = visualOdometry->pose.covariance[8];

            // time
            visual_dt = visualTimeCurrent - visualTimeLast;

            // header
            double timediff = ros::Time::now().toSec() - timeV + visualTimeCurrent;
            headerV = visualOdometry->header;
            headerV.stamp = ros::Time().fromSec(timediff);
            
            // compute average intensity
            averageIntensity = (averageIntensity1 + averageIntensity2)/2;

            //New measure
            filter.correction_visual_data(visualMeasure, E_visual, visual_dt, averageIntensity);
        }
    }

    void visualOdometryDHandler(const nav_msgs::Odometry::ConstPtr& visualOdometry){
        if (enableFilter && enableVisual && camera_type == 2){
            Eigen::MatrixXd E_visual(6,6);

            if (visualOdometry->pose.covariance[0] >= 9999.0 && visualOdometry->pose.pose.position.x == 0 && visualOdometry->pose.pose.position.y == 0 && visualOdometry->pose.pose.position.z == 0){
                // reset pose 
                rtabmap_msgs::ResetPose poseRgb;
                poseRgb.request.x = visualOdometry->pose.pose.position.x;
                poseRgb.request.y = visualOdometry->pose.pose.position.y;
                poseRgb.request.z = visualOdometry->pose.pose.position.z;

                double roll, pitch, yaw;
                geometry_msgs::Quaternion orientation = visualOdometry->pose.pose.orientation;
                tf::Matrix3x3(tf::Quaternion(orientation.x, orientation.y, orientation.z, orientation.w)).getRPY(roll, pitch, yaw);

                poseRgb.request.roll = roll;
                poseRgb.request.pitch = pitch;
                poseRgb.request.yaw = yaw;

                // call service
                srv_client_rgbd.call(poseRgb);
            }else{
                double timeV = ros::Time::now().toSec();

                if (visualActivated){
                    visualTimeLast = visualTimeCurrent;
                    visualTimeCurrent = visualOdometry->header.stamp.toSec();
                }else{
                    visualTimeCurrent = visualOdometry->header.stamp.toSec();
                    visualTimeLast = visualTimeCurrent + 0.01;
                    visualActivated = true;
                }  
                
                // roll, pitch and yaw 
                double roll, pitch, yaw;
                geometry_msgs::Quaternion orientation = visualOdometry->pose.pose.orientation;
                tf::Matrix3x3(tf::Quaternion(orientation.x, orientation.y, orientation.z, orientation.w)).getRPY(roll, pitch, yaw);

                visualMeasure.block(0,0,3,1) << visualOdometry->pose.pose.position.x, visualOdometry->pose.pose.position.y, visualOdometry->pose.pose.position.z;
                visualMeasure.block(3,0,3,1) << roll, pitch, yaw;    

                // covariance
                E_visual(0,0) = visualOdometry->pose.covariance[0];
                E_visual(0,1) = visualOdometry->pose.covariance[1];
                E_visual(0,2) = visualOdometry->pose.covariance[2];
                E_visual(1,0) = visualOdometry->pose.covariance[3];
                E_visual(1,1) = visualOdometry->pose.covariance[4];
                E_visual(1,2) = visualOdometry->pose.covariance[5];
                E_visual(2,0) = visualOdometry->pose.covariance[6];
                E_visual(2,1) = visualOdometry->pose.covariance[7];
                E_visual(2,2) = visualOdometry->pose.covariance[8];

                // time
                visual_dt = visualTimeCurrent - visualTimeLast;
                // visual_dt = 0.05;

                // header
                double timediff = ros::Time::now().toSec() - timeV + visualTimeCurrent;
                headerV = visualOdometry->header;
                headerV.stamp = ros::Time().fromSec(timediff);
                
                //New measure
                filter.correction_visual_data(visualMeasure, E_visual, visual_dt, averageIntensity);
            }            
        }
    }

    void camLeftHandler(const sensor_msgs::ImageConstPtr& camIn){
        int width = camIn->width;
        int height = camIn->height;

        int numPixels = width * height;
        double intensitySum = 0.0;

        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                int pixelIndex = y * width + x;
                uint8_t intensity = camIn->data[pixelIndex];

                intensitySum += intensity;
            }
        }

        averageIntensity2 = intensitySum / numPixels;
    }

    void camRightHandler(const sensor_msgs::ImageConstPtr& camIn){
        int width = camIn->width;
        int height = camIn->height;

        int numPixels = width * height;
        double intensitySum = 0.0;

        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                int pixelIndex = y * width + x;
                uint8_t intensity = camIn->data[pixelIndex];

                intensitySum += intensity;
            }
        }

        averageIntensity1 = intensitySum / numPixels;
    }

    void camRgbHandler(const sensor_msgs::ImageConstPtr& camIn){
        int width = camIn->width;
        int height = camIn->height;
        int numPixels = width * height;

        // Calculate pixel step size
        size_t pixel_step = camIn->step / camIn->width;

        // Define weights for RGB channels
        double redWeight = 0.2989;
        double greenWeight = 0.5870;
        double blueWeight = 0.1140;
        double intensitySum = 0.0;


        // Loop through image data and compute intensity
        for (size_t y = 0; y < camIn->height; ++y){
            for (size_t x = 0; x < camIn->width; ++x){
                size_t index = y * camIn->step + x * pixel_step;

                // Access RGB values
                uint8_t r = camIn->data[index];
                uint8_t g = camIn->data[index + 1];
                uint8_t b = camIn->data[index + 2];

                // Compute intensity using weighted average of RGB values
                intensitySum += redWeight * double(r) + greenWeight * double(g) + blueWeight * double(b);
            }
        }

        averageIntensity = intensitySum / numPixels;
    }

    void standard_pcl_cbk(const sensor_msgs::PointCloud2ConstPtr& msg){
        mtx_buffer.lock();
        scan_count ++;
        double preprocess_start_time = omp_get_wtime();
        if (msg->header.stamp.toSec() < last_timestamp_lidar)
        {
            ROS_ERROR("lidar loop back, clear buffer");
            lidar_buffer.clear();
        }

        PointCloudXYZI::Ptr  ptr(new PointCloudXYZI());
        p_pre->process(msg, ptr);
        lidar_buffer.push_back(ptr);
        time_buffer.push_back(msg->header.stamp.toSec());
        last_timestamp_lidar = msg->header.stamp.toSec();
        s_plot11[scan_count] = omp_get_wtime() - preprocess_start_time;
        mtx_buffer.unlock();
        sig_buffer.notify_all();
    }

    void livox_pcl_cbk(const livox_ros_driver::CustomMsgConstPtr& msg){
        mtx_buffer.lock();
        double preprocess_start_time = omp_get_wtime();
        scan_count ++;
        if (msg->header.stamp.toSec() < last_timestamp_lidar)
        {
            ROS_ERROR("lidar loop back, clear buffer");
            lidar_buffer.clear();
        }
        last_timestamp_lidar = msg->header.stamp.toSec();
        
        if (!time_sync_en && abs(last_timestamp_imu - last_timestamp_lidar) > 10.0 && !imu_buffer.empty() && !lidar_buffer.empty() )
        {
            printf("IMU and LiDAR not Synced, IMU time: %lf, lidar header time: %lf \n",last_timestamp_imu, last_timestamp_lidar);
        }

        if (time_sync_en && !timediff_set_flg && abs(last_timestamp_lidar - last_timestamp_imu) > 1 && !imu_buffer.empty())
        {
            timediff_set_flg = true;
            timediff_lidar_wrt_imu = last_timestamp_lidar + 0.1 - last_timestamp_imu;
            printf("Self sync IMU and LiDAR, time diff is %.10lf \n", timediff_lidar_wrt_imu);
        }

        PointCloudXYZI::Ptr  ptr(new PointCloudXYZI());
        p_pre->process(msg, ptr);
        lidar_buffer.push_back(ptr);
        time_buffer.push_back(last_timestamp_lidar);
        
        s_plot11[scan_count] = omp_get_wtime() - preprocess_start_time;
        mtx_buffer.unlock();
        sig_buffer.notify_all();
    }

    void imu_cbk(const sensor_msgs::ImuConstPtr& imuIn){
        publish_count ++;
        sensor_msgs::Imu::Ptr msg(new sensor_msgs::Imu(*imuIn));

        msg->header.stamp = ros::Time().fromSec(imuIn->header.stamp.toSec() - time_diff_lidar_to_imu);
        if (abs(timediff_lidar_wrt_imu) > 0.1 && time_sync_en)
        {
            msg->header.stamp = \
            ros::Time().fromSec(timediff_lidar_wrt_imu + imuIn->header.stamp.toSec());
        }

        double timestamp = msg->header.stamp.toSec();

        mtx_buffer.lock();

        if (timestamp < last_timestamp_imu)
        {
            ROS_WARN("imu loop back, clear buffer");
            imu_buffer.clear();
        }

        last_timestamp_imu = timestamp;

        imu_buffer.push_back(msg);
        mtx_buffer.unlock();
        sig_buffer.notify_all();

        // adaptive filter below
        double timeL = ros::Time::now().toSec();

        if (imuActivated){
            imuTimeLast = imuTimeCurrent;
            imuTimeCurrent = imuIn->header.stamp.toSec();
        }else{
            imuTimeCurrent = imuIn->header.stamp.toSec();
            imuTimeLast = imuTimeCurrent + 0.001;
            imuActivated = true;
        }       

        // roll, pitch and yaw 
        double roll, pitch, yaw;
        geometry_msgs::Quaternion orientation = imuIn->orientation;
        tf::Matrix3x3(tf::Quaternion(orientation.x, orientation.y, orientation.z, orientation.w)).getRPY(roll, pitch, yaw);

        // measure
        imuMeasure.block(0,0,3,1) << imuIn->linear_acceleration.x, imuIn->linear_acceleration.y, imuIn->linear_acceleration.z;
        imuMeasure.block(3,0,3,1) << imuIn->angular_velocity.x, imuIn->angular_velocity.y, imuIn->angular_velocity.z; 
        imuMeasure.block(6,0,3,1) << roll, pitch, yaw;

        // covariance
        E_imu.block(0,0,3,3) << imuIn->linear_acceleration_covariance[0], imuIn->linear_acceleration_covariance[1], imuIn->linear_acceleration_covariance[2],
                                imuIn->linear_acceleration_covariance[3], imuIn->linear_acceleration_covariance[4], imuIn->linear_acceleration_covariance[5],
                                imuIn->linear_acceleration_covariance[6], imuIn->linear_acceleration_covariance[7], imuIn->linear_acceleration_covariance[8];
        E_imu.block(3,3,3,3) << imuIn->angular_velocity_covariance[0], imuIn->angular_velocity_covariance[1], imuIn->angular_velocity_covariance[2],
                                imuIn->angular_velocity_covariance[3], imuIn->angular_velocity_covariance[4], imuIn->angular_velocity_covariance[5],
                                imuIn->angular_velocity_covariance[6], imuIn->angular_velocity_covariance[7], imuIn->angular_velocity_covariance[8];
        E_imu.block(6,6,3,3) << imuIn->orientation_covariance[0], imuIn->orientation_covariance[1], imuIn->orientation_covariance[2],
                                imuIn->orientation_covariance[3], imuIn->orientation_covariance[4], imuIn->orientation_covariance[5],
                                imuIn->orientation_covariance[6], imuIn->orientation_covariance[7], imuIn->orientation_covariance[8];

        E_imu.block(6,6,3,3) = imuG*E_imu.block(6,6,3,3);

        // time
        imu_dt = imuTimeCurrent - imuTimeLast;
        imu_dt = 0.01;

        // header
        double timediff = ros::Time::now().toSec() - timeL + imuTimeCurrent;
        headerI = imuIn->header;
        headerI.stamp = ros::Time().fromSec(timediff);

        // correction stage aqui
        filter.correction_imu_data(imuMeasure, E_imu, imu_dt);
    }

    //---------------------------
    // Fast-Lio2 functions
    //---------------------------
    static void h_share_model(state_ikfom &s, esekfom::dyn_share_datastruct<double> &ekfom_data);
    
    static void SigHandle(int sig);

    bool sync_packages(MeasureGroup &meas){
        if (lidar_buffer.empty() || imu_buffer.empty()) {
            return false;
        }

        /*** push a lidar scan ***/
        if(!lidar_pushed)
        {
            meas.lidar = lidar_buffer.front();
            meas.lidar_beg_time = time_buffer.front();


            if (meas.lidar->points.size() <= 1) // time too little
            {
                lidar_end_time = meas.lidar_beg_time + lidar_mean_scantime;
                ROS_WARN("Too few input point cloud!\n");
            }
            else if (meas.lidar->points.back().curvature / double(1000) < 0.5 * lidar_mean_scantime)
            {
                lidar_end_time = meas.lidar_beg_time + lidar_mean_scantime;
            }
            else
            {
                scan_num ++;
                lidar_end_time = meas.lidar_beg_time + meas.lidar->points.back().curvature / double(1000);
                lidar_mean_scantime += (meas.lidar->points.back().curvature / double(1000) - lidar_mean_scantime) / scan_num;
            }
            if(lidar_type == MARSIM)
                lidar_end_time = meas.lidar_beg_time;

            meas.lidar_end_time = lidar_end_time;

            lidar_pushed = true;
        }

        if (last_timestamp_imu < lidar_end_time)
        {
            return false;
        }

        /*** push imu data, and pop from imu buffer ***/
        double imu_time = imu_buffer.front()->header.stamp.toSec();
        meas.imu.clear();
        while ((!imu_buffer.empty()) && (imu_time < lidar_end_time))
        {
            imu_time = imu_buffer.front()->header.stamp.toSec();
            if(imu_time > lidar_end_time) break;
            meas.imu.push_back(imu_buffer.front());
            imu_buffer.pop_front();
        }

        lidar_buffer.pop_front();
        time_buffer.pop_front();
        lidar_pushed = false;
        return true;
    }

    inline void dump_lio_state_to_log(FILE *fp){
        V3D rot_ang(Log(state_point.rot.toRotationMatrix()));
        fprintf(fp, "%lf ", Measures.lidar_beg_time - first_lidar_time);
        fprintf(fp, "%lf %lf %lf ", rot_ang(0), rot_ang(1), rot_ang(2));                   // Angle
        fprintf(fp, "%lf %lf %lf ", state_point.pos(0), state_point.pos(1), state_point.pos(2)); // Pos  
        fprintf(fp, "%lf %lf %lf ", 0.0, 0.0, 0.0);                                        // omega  
        fprintf(fp, "%lf %lf %lf ", state_point.vel(0), state_point.vel(1), state_point.vel(2)); // Vel  
        fprintf(fp, "%lf %lf %lf ", 0.0, 0.0, 0.0);                                        // Acc  
        fprintf(fp, "%lf %lf %lf ", state_point.bg(0), state_point.bg(1), state_point.bg(2));    // Bias_g  
        fprintf(fp, "%lf %lf %lf ", state_point.ba(0), state_point.ba(1), state_point.ba(2));    // Bias_a  
        fprintf(fp, "%lf %lf %lf ", state_point.grav[0], state_point.grav[1], state_point.grav[2]); // Bias_a  
        fprintf(fp, "\r\n");  
        fflush(fp);
    }

    void points_cache_collect(){
        PointVector points_history;
        ikdtree.acquire_removed_points(points_history);
        // for (int i = 0; i < points_history.size(); i++) _featsArray->push_back(points_history[i]);
    }

    void pointBodyToWorld(PointType const * const pi, PointType * const po){
        V3D p_body(pi->x, pi->y, pi->z);
        V3D p_global(state_point.rot * (state_point.offset_R_L_I*p_body + state_point.offset_T_L_I) + state_point.pos);

        po->x = p_global(0);
        po->y = p_global(1);
        po->z = p_global(2);
        po->intensity = pi->intensity;
    }

    template<typename T>
    void pointBodyToWorld(const Matrix<T, 3, 1> &pi, Matrix<T, 3, 1> &po){
        V3D p_body(pi[0], pi[1], pi[2]);
        V3D p_global(state_point.rot * (state_point.offset_R_L_I*p_body + state_point.offset_T_L_I) + state_point.pos);

        po[0] = p_global(0);
        po[1] = p_global(1);
        po[2] = p_global(2);
    }

    void pointBodyToWorldFilter(PointType const * const pi, PointType * const po){
        float x1 = cYaw * pi->x - sYaw * pi->y;
        float y1 = sYaw * pi->x + cYaw * pi->y;
        float z1 = pi->z;

        float x2 = x1;
        float y2 = cRoll * y1 - sRoll * z1;
        float z2 = sRoll * y1 + cRoll * z1;

        po->x = cPitch * x2 + sPitch * z2 + tX;
        po->y = y2 + tY;
        po->z = -sPitch * x2 + cPitch * z2 + tZ;
        po->intensity = pi->intensity;
    }

    void RGBpointBodyToWorld(PointType const * const pi, PointType * const po){
        V3D p_body(pi->x, pi->y, pi->z);
        V3D p_global(state_point.rot * (state_point.offset_R_L_I*p_body + state_point.offset_T_L_I) + state_point.pos);

        po->x = p_global(0);
        po->y = p_global(1);
        po->z = p_global(2);
        po->intensity = pi->intensity;
    }

    void RGBpointBodyLidarToIMU(PointType const * const pi, PointType * const po){
        V3D p_body_lidar(pi->x, pi->y, pi->z);
        V3D p_body_imu(state_point.offset_R_L_I*p_body_lidar + state_point.offset_T_L_I);

        po->x = p_body_imu(0);
        po->y = p_body_imu(1);
        po->z = p_body_imu(2);
        po->intensity = pi->intensity;
    }

    template<typename T>
    void set_posestamp(T & out){
        out.pose.position.x = state_point.pos(0);
        out.pose.position.y = state_point.pos(1);
        out.pose.position.z = state_point.pos(2);
        out.pose.orientation.x = geoQuat.x;
        out.pose.orientation.y = geoQuat.y;
        out.pose.orientation.z = geoQuat.z;
        out.pose.orientation.w = geoQuat.w;
        
    }

    void lasermap_fov_segment(){
        cub_needrm.clear();
        kdtree_delete_counter = 0;
        kdtree_delete_time = 0.0;    
        pointBodyToWorld(XAxisPoint_body, XAxisPoint_world);
        V3D pos_LiD = pos_lid;
        if (!Localmap_Initialized){
            for (int i = 0; i < 3; i++){
                LocalMap_Points.vertex_min[i] = pos_LiD(i) - cube_len / 2.0;
                LocalMap_Points.vertex_max[i] = pos_LiD(i) + cube_len / 2.0;
            }
            Localmap_Initialized = true;
            return;
        }
        float dist_to_map_edge[3][2];
        bool need_move = false;
        for (int i = 0; i < 3; i++){
            dist_to_map_edge[i][0] = fabs(pos_LiD(i) - LocalMap_Points.vertex_min[i]);
            dist_to_map_edge[i][1] = fabs(pos_LiD(i) - LocalMap_Points.vertex_max[i]);
            if (dist_to_map_edge[i][0] <= MOV_THRESHOLD * DET_RANGE || dist_to_map_edge[i][1] <= MOV_THRESHOLD * DET_RANGE) need_move = true;
        }
        if (!need_move) return;
        BoxPointType New_LocalMap_Points, tmp_boxpoints;
        New_LocalMap_Points = LocalMap_Points;
        float mov_dist = max((cube_len - 2.0 * MOV_THRESHOLD * DET_RANGE) * 0.5 * 0.9, double(DET_RANGE * (MOV_THRESHOLD -1)));
        for (int i = 0; i < 3; i++){
            tmp_boxpoints = LocalMap_Points;
            if (dist_to_map_edge[i][0] <= MOV_THRESHOLD * DET_RANGE){
                New_LocalMap_Points.vertex_max[i] -= mov_dist;
                New_LocalMap_Points.vertex_min[i] -= mov_dist;
                tmp_boxpoints.vertex_min[i] = LocalMap_Points.vertex_max[i] - mov_dist;
                cub_needrm.push_back(tmp_boxpoints);
            } else if (dist_to_map_edge[i][1] <= MOV_THRESHOLD * DET_RANGE){
                New_LocalMap_Points.vertex_max[i] += mov_dist;
                New_LocalMap_Points.vertex_min[i] += mov_dist;
                tmp_boxpoints.vertex_max[i] = LocalMap_Points.vertex_min[i] + mov_dist;
                cub_needrm.push_back(tmp_boxpoints);
            }
        }
        LocalMap_Points = New_LocalMap_Points;

        points_cache_collect();
        double delete_begin = omp_get_wtime();
        if(cub_needrm.size() > 0) kdtree_delete_counter = ikdtree.Delete_Point_Boxes(cub_needrm);
        kdtree_delete_time = omp_get_wtime() - delete_begin;
    }

    void updatePointAssociateToMapSinCos(){
        // orientation
        double roll, pitch, yaw;
        tf::Matrix3x3(tf::Quaternion(state_point_map.rot.coeffs()[0], state_point_map.rot.coeffs()[1], state_point_map.rot.coeffs()[2], state_point_map.rot.coeffs()[3])).getRPY(roll, pitch, yaw);
                
        cRoll = cos(roll);
        sRoll = sin(roll);

        cPitch = cos(pitch);
        sPitch = sin(pitch);

        cYaw = cos(yaw);
        sYaw = sin(yaw);

        tX = state_point_map.pos(0);
        tY = state_point_map.pos(1);
        tZ = state_point_map.pos(2);
    }

    void map_incremental(){
        PointVector PointToAdd;
        PointVector PointNoNeedDownsample;
        PointToAdd.reserve(feats_down_size);
        PointNoNeedDownsample.reserve(feats_down_size);

        // aqui
        if (enableFilter){
            updatePointAssociateToMapSinCos();
        }        

        for (int i = 0; i < feats_down_size; i++)
        {
            /* transform to world frame - change here - create another function to use the filtered pose */ 
            if (enableFilter){
                pointBodyToWorldFilter(&(feats_down_body->points[i]), &(feats_down_world->points[i]));
            }else{
                pointBodyToWorld(&(feats_down_body->points[i]), &(feats_down_world->points[i]));
            }
                        
            /* decide if need add to map */
            if (!Nearest_Points[i].empty() && flg_EKF_inited)
            {
                const PointVector &points_near = Nearest_Points[i];
                bool need_add = true;
                BoxPointType Box_of_Point;
                PointType downsample_result, mid_point; 
                mid_point.x = floor(feats_down_world->points[i].x/filter_size_map_min)*filter_size_map_min + 0.5 * filter_size_map_min;
                mid_point.y = floor(feats_down_world->points[i].y/filter_size_map_min)*filter_size_map_min + 0.5 * filter_size_map_min;
                mid_point.z = floor(feats_down_world->points[i].z/filter_size_map_min)*filter_size_map_min + 0.5 * filter_size_map_min;
                float dist  = calc_dist(feats_down_world->points[i],mid_point);
                if (fabs(points_near[0].x - mid_point.x) > 0.5 * filter_size_map_min && fabs(points_near[0].y - mid_point.y) > 0.5 * filter_size_map_min && fabs(points_near[0].z - mid_point.z) > 0.5 * filter_size_map_min){
                    PointNoNeedDownsample.push_back(feats_down_world->points[i]);
                    continue;
                }
                for (int readd_i = 0; readd_i < NUM_MATCH_POINTS; readd_i ++)
                {
                    if (points_near.size() < NUM_MATCH_POINTS) break;
                    if (calc_dist(points_near[readd_i], mid_point) < dist)
                    {
                        need_add = false;
                        break;
                    }
                }
                if (need_add) PointToAdd.push_back(feats_down_world->points[i]);
            }
            else
            {
                PointToAdd.push_back(feats_down_world->points[i]);
            }
        }

        double st_time = omp_get_wtime();
        add_point_size = ikdtree.Add_Points(PointToAdd, true);
        ikdtree.Add_Points(PointNoNeedDownsample, false); 
        add_point_size = PointToAdd.size() + PointNoNeedDownsample.size();
        kdtree_incremental_time = omp_get_wtime() - st_time;
    }

    void save_map(){
        /**************** save map ****************/
        /* 1. make sure you have enough memories
        /* 2. pcd save will largely influence the real-time performences **/
        if (pcl_wait_save->size() > 0 && pcd_save_en)
        {
            string file_name = string("scans.pcd");
            string all_points_dir(string(string(ROOT_DIR) + "PCD/") + file_name);
            pcl::PCDWriter pcd_writer;
            cout << "current scan saved to /PCD/" << file_name<<endl;
            pcd_writer.writeBinary(all_points_dir, *pcl_wait_save);
        }

        fout_out.close();
        fout_pre.close();

        if (runtime_pos_log)
        {
            vector<double> t, s_vec, s_vec2, s_vec3, s_vec4, s_vec5, s_vec6, s_vec7;    
            FILE *fp2;
            string log_dir = root_dir + "/Log/fast_lio_time_log.csv";
            fp2 = fopen(log_dir.c_str(),"w");
            fprintf(fp2,"time_stamp, total time, scan point size, incremental time, search time, delete size, delete time, tree size st, tree size end, add point size, preprocess time\n");
            for (int i = 0;i<time_log_counter; i++){
                fprintf(fp2,"%0.8f,%0.8f,%d,%0.8f,%0.8f,%d,%0.8f,%d,%d,%d,%0.8f\n",T1[i],s_plot[i],int(s_plot2[i]),s_plot3[i],s_plot4[i],int(s_plot5[i]),s_plot6[i],int(s_plot7[i]),int(s_plot8[i]), int(s_plot10[i]), s_plot11[i]);
                t.push_back(T1[i]);
                s_vec.push_back(s_plot9[i]);
                s_vec2.push_back(s_plot3[i] + s_plot6[i]);
                s_vec3.push_back(s_plot4[i]);
                s_vec5.push_back(s_plot[i]);
            }
            fclose(fp2);
        }
    }

    //------------------
    // Adaptive Filter functions
    //------------------
    void filter_start(){
        ROS_INFO("\033[1;32mAdaptive Filter:\033[0m Filter Started.");
        filter.start();
    }

    void filter_stop(){
        filter.stop();
    }

    //-----------------------------
    // Fast-Lio2 Run
    //-----------------------------
    void run(){
        signal(SIGINT, SigHandle);
        ros::Rate rate(5000);
        while (ros::ok()){
            if (flg_exit) break;
            if(sync_packages(Measures)){
                if (flg_first_scan)
                {
                    first_lidar_time = Measures.lidar_beg_time;
                    p_imu->first_lidar_time = first_lidar_time;
                    flg_first_scan = false;
                    continue;
                }

                double t0,t1,t2,t3,t4,t5,match_start, solve_start, svd_time;

                match_time = 0;
                kdtree_search_time = 0.0;
                solve_time = 0;
                solve_const_H_time = 0;
                svd_time   = 0;
                t0 = omp_get_wtime();

                p_imu->Process(Measures, kf, feats_undistort);
                state_point = kf.get_x();
                pos_lid = state_point.pos + state_point.rot * state_point.offset_T_L_I;

                if (feats_undistort->empty() || (feats_undistort == NULL))
                {
                    ROS_WARN("No point, skip this scan!\n");
                    continue;
                }

                flg_EKF_inited = (Measures.lidar_beg_time - first_lidar_time) < INIT_TIME ? \
                            false : true;
                
                /*** Segment the map in lidar FOV ***/
                lasermap_fov_segment();

                /*** downsample the feature points in a scan ***/
                downSizeFilterSurf.setInputCloud(feats_undistort);
                downSizeFilterSurf.filter(*feats_down_body);
                t1 = omp_get_wtime();
                feats_down_size = feats_down_body->points.size();

                /*** initialize the map kdtree - inserir o mapa aqui***/
                if(ikdtree.Root_Node == nullptr)
                {
                    if(feats_down_size > 5)
                    {
                        ikdtree.set_downsample_param(filter_size_map_min);
                        feats_down_world->resize(feats_down_size);
                        for(int i = 0; i < feats_down_size; i++)
                        {
                            pointBodyToWorld(&(feats_down_body->points[i]), &(feats_down_world->points[i]));
                        }
                        ikdtree.Build(feats_down_world->points);
                    }
                    continue;
                }
                int featsFromMapNum = ikdtree.validnum();
                kdtree_size_st = ikdtree.size();
                
                // cout<<"[ mapping ]: In num: "<<feats_undistort->points.size()<<" downsamp "<<feats_down_size<<" Map num: "<<featsFromMapNum<<"effect num:"<<effct_feat_num<<endl;

                /*** ICP and iterated Kalman filter update ***/
                if (feats_down_size < 5)
                {
                    ROS_WARN("No point, skip this scan!\n");
                    continue;
                }
                
                normvec->resize(feats_down_size);
                feats_down_world->resize(feats_down_size);

                V3D ext_euler = SO3ToEuler(state_point.offset_R_L_I);
                fout_pre<<setw(20)<<Measures.lidar_beg_time - first_lidar_time<<" "<<euler_cur.transpose()<<" "<< state_point.pos.transpose()<<" "<<ext_euler.transpose() << " "<<state_point.offset_T_L_I.transpose()<< " " << state_point.vel.transpose() \
                <<" "<<state_point.bg.transpose()<<" "<<state_point.ba.transpose()<<" "<<state_point.grav<< endl;

                if(publish_map) // If you need to see map point, change to "if(1)"
                {
                    PointVector ().swap(ikdtree.PCL_Storage);
                    ikdtree.flatten(ikdtree.Root_Node, ikdtree.PCL_Storage, NOT_RECORD);
                    featsFromMap->clear();
                    featsFromMap->points = ikdtree.PCL_Storage;
                }

                pointSearchInd_surf.resize(feats_down_size);
                Nearest_Points.resize(feats_down_size);
                int  rematch_num = 0;
                bool nearest_search_en = true; //

                t2 = omp_get_wtime();
                
                /*** iterated state estimation ***/
                double t_update_start = omp_get_wtime();
                double solve_H_time = 0;
                kf.update_iterated_dyn_share_modified(LASER_POINT_COV, solve_H_time);
                state_point = kf.get_x();

                /***************Insert the adapttive filter here **********************/
                // update state_point or create a filtered pose to increment the map or the undistort the point cloud as mentioned before in the code.
                if (enableFilter){
                    // auto state_p = kf.get_x();
                    auto P = kf.get_P();

                    // measurement
                    Eigen::VectorXd lidarMeasure(6), X_out(12);
                    Eigen::MatrixXd E_lidar(6,6), E_out(12,12);

                    // postion
                    lidarMeasure.block(0,0,3,1) << state_point.pos(0), state_point.pos(1), state_point.pos(2);  
                    // orientation
                    double roll, pitch, yaw;
                    tf::Matrix3x3(tf::Quaternion(state_point_map.rot.coeffs()[0], state_point_map.rot.coeffs()[1], state_point_map.rot.coeffs()[2], state_point_map.rot.coeffs()[3])).getRPY(roll, pitch, yaw);
                    // auto euler_c = SO3ToEuler(state_point.rot); // eigen??
                    lidarMeasure.block(3,0,3,1) << roll, pitch, yaw;
                    // covariancia
                    E_lidar = P;

                    // time
                    if (firstLidarPublish){
                        lidarTimeLast = lidarTimeCurrent;
                        lidarTimeCurrent = lidar_end_time;
                    }else{
                        lidarTimeCurrent = lidar_end_time;
                        lidarTimeLast = lidarTimeCurrent + 0.01;
                        firstLidarPublish = true;
                    }
                    
                    // time diff
                    lidar_dt = lidarTimeCurrent - lidarTimeLast;

                    // correction stage
                    filter.correction_lidar_data(lidarMeasure, E_lidar, lidar_dt, 0.0, 0.0); // parei aqui. adicionar flag para publicação??

                    // get state here
                    filter.get_state(X_out, E_out);

                    // change fast-lio2 results
                    state_point_map.pos(0) = X_out(0);
                    state_point_map.pos(1) = X_out(1);
                    state_point_map.pos(2) = X_out(2);
                    tf::Quaternion q;
                    q.setRPY(X_out(3), X_out(4), X_out(5));
                    state_point_map.rot.coeffs()[0] = q.x();
                    state_point_map.rot.coeffs()[1] = q.y();
                    state_point_map.rot.coeffs()[2] = q.z();
                    state_point_map.rot.coeffs()[3] = q.w();

                    publish_odometry_filter(pubFilteredOdometry, E_out);
                }
                
                // publisher - normal

                euler_cur = SO3ToEuler(state_point.rot);
                pos_lid = state_point.pos + state_point.rot * state_point.offset_T_L_I;
                geoQuat.x = state_point.rot.coeffs()[0];
                geoQuat.y = state_point.rot.coeffs()[1];
                geoQuat.z = state_point.rot.coeffs()[2];
                geoQuat.w = state_point.rot.coeffs()[3];

                double t_update_end = omp_get_wtime();

                /******* Publish odometry *******/
                publish_odometry(pubOdomAftMapped);

                /*** add the feature points to map kdtree ***/
                t3 = omp_get_wtime();
                map_incremental();
                t5 = omp_get_wtime();
                
                /******* Publish points *******/
                if (path_en)                         publish_path(pubPath);
                if (scan_pub_en || pcd_save_en)      publish_frame_world(pubLaserCloudFull);
                if (scan_pub_en && scan_body_pub_en) publish_frame_body(pubLaserCloudFull_body);
                // publish_effect_world(pubLaserCloudEffect);
                if (publish_map){
                    publish_cloud_map(pubLaserCloudMap);
                }
                
                /*** Debug variables ***/
                if (runtime_pos_log)
                {
                    frame_num ++;
                    kdtree_size_end = ikdtree.size();
                    aver_time_consu = aver_time_consu * (frame_num - 1) / frame_num + (t5 - t0) / frame_num;
                    aver_time_icp = aver_time_icp * (frame_num - 1)/frame_num + (t_update_end - t_update_start) / frame_num;
                    aver_time_match = aver_time_match * (frame_num - 1)/frame_num + (match_time)/frame_num;
                    aver_time_incre = aver_time_incre * (frame_num - 1)/frame_num + (kdtree_incremental_time)/frame_num;
                    aver_time_solve = aver_time_solve * (frame_num - 1)/frame_num + (solve_time + solve_H_time)/frame_num;
                    aver_time_const_H_time = aver_time_const_H_time * (frame_num - 1)/frame_num + solve_time / frame_num;
                    T1[time_log_counter] = Measures.lidar_beg_time;
                    s_plot[time_log_counter] = t5 - t0;
                    s_plot2[time_log_counter] = feats_undistort->points.size();
                    s_plot3[time_log_counter] = kdtree_incremental_time;
                    s_plot4[time_log_counter] = kdtree_search_time;
                    s_plot5[time_log_counter] = kdtree_delete_counter;
                    s_plot6[time_log_counter] = kdtree_delete_time;
                    s_plot7[time_log_counter] = kdtree_size_st;
                    s_plot8[time_log_counter] = kdtree_size_end;
                    s_plot9[time_log_counter] = aver_time_consu;
                    s_plot10[time_log_counter] = add_point_size;
                    time_log_counter ++;
                    printf("[ mapping ]: time: IMU + Map + Input Downsample: %0.6f ave match: %0.6f ave solve: %0.6f  ave ICP: %0.6f  map incre: %0.6f ave total: %0.6f icp: %0.6f construct H: %0.6f \n",t1-t0,aver_time_match,aver_time_solve,t3-t1,t5-t3,aver_time_consu,aver_time_icp, aver_time_const_H_time);
                    ext_euler = SO3ToEuler(state_point.offset_R_L_I);
                    fout_out << setw(20) << Measures.lidar_beg_time - first_lidar_time << " " << euler_cur.transpose() << " " << state_point.pos.transpose()<< " " << ext_euler.transpose() << " "<<state_point.offset_T_L_I.transpose()<<" "<< state_point.vel.transpose() \
                    <<" "<<state_point.bg.transpose()<<" "<<state_point.ba.transpose()<<" "<<state_point.grav<<" "<<feats_undistort->points.size()<<endl;
                    dump_lio_state_to_log(fp);
                }
            }

            rate.sleep();
        }

        // save map
        save_map();
    }

    //-------------------------
    // Publish functions
    //-------------------------
    void publish_odometry_filter(const ros::Publisher & pubFilteredOdometry, Eigen::MatrixXd Eo){
        nav_msgs::Odometry odomFilter;
        odomFilter.header.frame_id = "camera_init";
        odomFilter.child_frame_id = "body_filter";
        odomFilter.header.stamp = ros::Time().fromSec(lidar_end_time);// ros::Time().fromSec(lidar_end_time);

        // adicionar aqui
        odomFilter.pose.pose.position.x = state_point_map.pos(0);
        odomFilter.pose.pose.position.x = state_point_map.pos(0);
        odomFilter.pose.pose.position.x = state_point_map.pos(0);    
        odomFilter.pose.pose.orientation.x = state_point_map.rot.coeffs()[0];
        odomFilter.pose.pose.orientation.y = state_point_map.rot.coeffs()[1];
        odomFilter.pose.pose.orientation.z = state_point_map.rot.coeffs()[2];
        odomFilter.pose.pose.orientation.w = state_point_map.rot.coeffs()[3];
        
        pubFilteredOdometry.publish(odomFilter);

        static tf::TransformBroadcaster br;
        tf::Transform                   transform;
        tf::Quaternion                  q;
        transform.setOrigin(tf::Vector3(odomFilter.pose.pose.position.x, \
                                        odomFilter.pose.pose.position.y, \
                                        odomFilter.pose.pose.position.z));
        q.setW(odomFilter.pose.pose.orientation.w);
        q.setX(odomFilter.pose.pose.orientation.x);
        q.setY(odomFilter.pose.pose.orientation.y);
        q.setZ(odomFilter.pose.pose.orientation.z);
        transform.setRotation( q );
        br.sendTransform( tf::StampedTransform( transform, odomFilter.header.stamp, "camera_init", "body_filter" ) );
    }

    void publish_odometry(const ros::Publisher & pubOdomAftMapped){
        odomAftMapped.header.frame_id = "camera_init";
        odomAftMapped.child_frame_id = "body";
        odomAftMapped.header.stamp = ros::Time().fromSec(lidar_end_time);// ros::Time().fromSec(lidar_end_time);
        set_posestamp(odomAftMapped.pose);
        pubOdomAftMapped.publish(odomAftMapped);
        auto P = kf.get_P();
        for (int i = 0; i < 6; i ++)
        {
            int k = i < 3 ? i + 3 : i - 3;
            odomAftMapped.pose.covariance[i*6 + 0] = P(k, 3);
            odomAftMapped.pose.covariance[i*6 + 1] = P(k, 4);
            odomAftMapped.pose.covariance[i*6 + 2] = P(k, 5);
            odomAftMapped.pose.covariance[i*6 + 3] = P(k, 0);
            odomAftMapped.pose.covariance[i*6 + 4] = P(k, 1);
            odomAftMapped.pose.covariance[i*6 + 5] = P(k, 2);
        }

        static tf::TransformBroadcaster br;
        tf::Transform                   transform;
        tf::Quaternion                  q;
        transform.setOrigin(tf::Vector3(odomAftMapped.pose.pose.position.x, \
                                        odomAftMapped.pose.pose.position.y, \
                                        odomAftMapped.pose.pose.position.z));
        q.setW(odomAftMapped.pose.pose.orientation.w);
        q.setX(odomAftMapped.pose.pose.orientation.x);
        q.setY(odomAftMapped.pose.pose.orientation.y);
        q.setZ(odomAftMapped.pose.pose.orientation.z);
        transform.setRotation( q );
        br.sendTransform( tf::StampedTransform( transform, odomAftMapped.header.stamp, "camera_init", "body" ) );
    }

    void publish_path(const ros::Publisher pubPath){
        set_posestamp(msg_body_pose);
        msg_body_pose.header.stamp = ros::Time().fromSec(lidar_end_time);
        msg_body_pose.header.frame_id = "camera_init";

        /*** if path is too large, the rvis will crash ***/
        static int jjj = 0;
        jjj++;
        if (jjj % 10 == 0) 
        {
            path.poses.push_back(msg_body_pose);
            pubPath.publish(path);
        }
    }

    void publish_frame_world(const ros::Publisher & pubLaserCloudFull){
        if(scan_pub_en)
        {
            PointCloudXYZI::Ptr laserCloudFullRes(dense_pub_en ? feats_undistort : feats_down_body);
            int size = laserCloudFullRes->points.size();
            PointCloudXYZI::Ptr laserCloudWorld( \
                            new PointCloudXYZI(size, 1));

            for (int i = 0; i < size; i++)
            {
                RGBpointBodyToWorld(&laserCloudFullRes->points[i], \
                                    &laserCloudWorld->points[i]);
            }

            sensor_msgs::PointCloud2 laserCloudmsg;
            pcl::toROSMsg(*laserCloudWorld, laserCloudmsg);
            laserCloudmsg.header.stamp = ros::Time().fromSec(lidar_end_time);
            laserCloudmsg.header.frame_id = "camera_init";
            pubLaserCloudFull.publish(laserCloudmsg);
            publish_count -= PUBFRAME_PERIOD;
        }

        /**************** save map ****************/
        /* 1. make sure you have enough memories
        /* 2. noted that pcd save will influence the real-time performences **/
        if (pcd_save_en)
        {
            int size = feats_undistort->points.size();
            PointCloudXYZI::Ptr laserCloudWorld( \
                            new PointCloudXYZI(size, 1));

            for (int i = 0; i < size; i++)
            {
                RGBpointBodyToWorld(&feats_undistort->points[i], \
                                    &laserCloudWorld->points[i]);
            }
            *pcl_wait_save += *laserCloudWorld;

            static int scan_wait_num = 0;
            scan_wait_num ++;
            if (pcl_wait_save->size() > 0 && pcd_save_interval > 0  && scan_wait_num >= pcd_save_interval)
            {
                pcd_index ++;
                string all_points_dir(string(string(ROOT_DIR) + "PCD/scans_") + to_string(pcd_index) + string(".pcd"));
                pcl::PCDWriter pcd_writer;
                cout << "current scan saved to /PCD/" << all_points_dir << endl;
                pcd_writer.writeBinary(all_points_dir, *pcl_wait_save);
                pcl_wait_save->clear();
                scan_wait_num = 0;
            }
        }
    }

    void publish_frame_body(const ros::Publisher & pubLaserCloudFull_body){
        int size = feats_undistort->points.size();
        PointCloudXYZI::Ptr laserCloudIMUBody(new PointCloudXYZI(size, 1));

        for (int i = 0; i < size; i++)
        {
            RGBpointBodyLidarToIMU(&feats_undistort->points[i], \
                                &laserCloudIMUBody->points[i]);
        }

        sensor_msgs::PointCloud2 laserCloudmsg;
        pcl::toROSMsg(*laserCloudIMUBody, laserCloudmsg);
        laserCloudmsg.header.stamp = ros::Time().fromSec(lidar_end_time);
        laserCloudmsg.header.frame_id = "body";
        pubLaserCloudFull_body.publish(laserCloudmsg);
        publish_count -= PUBFRAME_PERIOD;
    }

    void publish_cloud_map(const ros::Publisher & pubLaserCloudMap){
        sensor_msgs::PointCloud2 laserCloudMap;
        pcl::toROSMsg(*featsFromMap, laserCloudMap);
        laserCloudMap.header.stamp = ros::Time().fromSec(lidar_end_time);
        laserCloudMap.header.frame_id = "camera_init";
        pubLaserCloudMap.publish(laserCloudMap);
    }

};

//----------------------------
// static member definition
//----------------------------
fastLioMapping* fastLioMapping::instance = nullptr;

void fastLioMapping::h_share_model(state_ikfom &s, esekfom::dyn_share_datastruct<double> &ekfom_data){
    double match_start = omp_get_wtime();
    instance->laserCloudOri->clear(); 
    instance->corr_normvect->clear(); 
    instance->total_residual = 0.0; 

    /** closest surface search and residual computation **/
    #ifdef MP_EN
        omp_set_num_threads(MP_PROC_NUM);
        #pragma omp parallel for
    #endif
    for (int i = 0; i < instance->feats_down_size; i++)
    {
        PointType &point_body  = instance->feats_down_body->points[i]; 
        PointType &point_world = instance->feats_down_world->points[i]; 

        /* transform to world frame */
        V3D p_body(point_body.x, point_body.y, point_body.z);
        V3D p_global(s.rot * (s.offset_R_L_I*p_body + s.offset_T_L_I) + s.pos);
        point_world.x = p_global(0);
        point_world.y = p_global(1);
        point_world.z = p_global(2);
        point_world.intensity = point_body.intensity;

        vector<float> pointSearchSqDis(NUM_MATCH_POINTS);

        auto &points_near = instance->Nearest_Points[i];

        if (ekfom_data.converge)
        {
            /** Find the closest surfaces in the map **/
            instance->ikdtree.Nearest_Search(point_world, NUM_MATCH_POINTS, points_near, pointSearchSqDis);
            instance->point_selected_surf[i] = points_near.size() < NUM_MATCH_POINTS ? false : pointSearchSqDis[NUM_MATCH_POINTS - 1] > 5 ? false : true;
        }

        if (!instance->point_selected_surf[i]) continue;

        VF(4) pabcd;
        instance->point_selected_surf[i] = false;
        if (esti_plane(pabcd, points_near, 0.1f))
        {
            float pd2 = pabcd(0) * point_world.x + pabcd(1) * point_world.y + pabcd(2) * point_world.z + pabcd(3);
            float s = 1 - 0.9 * fabs(pd2) / sqrt(p_body.norm());

            if (s > 0.9)
            {
                instance->point_selected_surf[i] = true;
                instance->normvec->points[i].x = pabcd(0);
                instance->normvec->points[i].y = pabcd(1);
                instance->normvec->points[i].z = pabcd(2);
                instance->normvec->points[i].intensity = pd2;
                instance->res_last[i] = abs(pd2);
            }
        }
    }
    
    instance->effct_feat_num = 0;

    for (int i = 0; i < instance->feats_down_size; i++)
    {
        if (instance->point_selected_surf[i])
        {
            instance->laserCloudOri->points[instance->effct_feat_num] = instance->feats_down_body->points[i];
            instance->corr_normvect->points[instance->effct_feat_num] = instance->normvec->points[i];
            instance->total_residual += instance->res_last[i];
            instance->effct_feat_num ++;
        }
    }

    if (instance->effct_feat_num < 1)
    {
        ekfom_data.valid = false;
        ROS_WARN("No Effective Points! \n");
        return;
    }

    instance->res_mean_last = instance->total_residual / instance->effct_feat_num;
    instance->match_time  += omp_get_wtime() - match_start;
    double solve_start_  = omp_get_wtime();
    
    /*** Computation of Measuremnt Jacobian matrix H and measurents vector ***/
    ekfom_data.h_x = MatrixXd::Zero(instance->effct_feat_num, 12); //23
    ekfom_data.h.resize(instance->effct_feat_num);

    for (int i = 0; i < instance->effct_feat_num; i++)
    {
        const PointType &laser_p  = instance->laserCloudOri->points[i];
        V3D point_this_be(laser_p.x, laser_p.y, laser_p.z);
        M3D point_be_crossmat;
        point_be_crossmat << SKEW_SYM_MATRX(point_this_be);
        V3D point_this = s.offset_R_L_I * point_this_be + s.offset_T_L_I;
        M3D point_crossmat;
        point_crossmat<<SKEW_SYM_MATRX(point_this);

        /*** get the normal vector of closest surface/corner ***/
        const PointType &norm_p = instance->corr_normvect->points[i];
        V3D norm_vec(norm_p.x, norm_p.y, norm_p.z);

        /*** calculate the Measuremnt Jacobian matrix H ***/
        V3D C(s.rot.conjugate() *norm_vec);
        V3D A(point_crossmat * C);
        if (instance->extrinsic_est_en)
        {
            V3D B(point_be_crossmat * s.offset_R_L_I.conjugate() * C); //s.rot.conjugate()*norm_vec);
            ekfom_data.h_x.block<1, 12>(i,0) << norm_p.x, norm_p.y, norm_p.z, VEC_FROM_ARRAY(A), VEC_FROM_ARRAY(B), VEC_FROM_ARRAY(C);
        }
        else
        {
            ekfom_data.h_x.block<1, 12>(i,0) << norm_p.x, norm_p.y, norm_p.z, VEC_FROM_ARRAY(A), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
        }

        /*** Measuremnt: distance to the closest surface/corner ***/
        ekfom_data.h(i) = -norm_p.intensity;
    }
    instance->solve_time += omp_get_wtime() - solve_start_;
}

void fastLioMapping::SigHandle(int sig){
    instance->flg_exit = true;
    ROS_WARN("catch sig %d", sig);
    instance->sig_buffer.notify_all();
}

//----------------------------
// Main
//----------------------------
int main(int argc, char** argv)
{
    ros::init(argc, argv, "fast_lio_mapping");

    fastLioMapping fastlio;

    ros::NodeHandle nh_;
    try
    {
        nh_.param("publish/path_en", fastlio.path_en, true);
        nh_.param("publish/scan_publish_en", fastlio.scan_pub_en, true);
        nh_.param("publish/dense_publish_en", fastlio.dense_pub_en, true);
        nh_.param("publish/scan_bodyframe_pub_en", fastlio.scan_body_pub_en, true);
        nh_.param("max_iteration", fastlio.NUM_MAX_ITERATIONS, int(4));
        nh_.param("map_file_path", fastlio.map_file_path, std::string(""));
        nh_.param("common/lid_topic", fastlio.lid_topic, std::string("/livox/lidar"));
        nh_.param("common/imu_topic", fastlio.imu_topic, std::string("/livox/imu"));
        nh_.param("common/time_sync_en", fastlio.time_sync_en, false);
        nh_.param("common/time_offset_lidar_to_imu", fastlio.time_diff_lidar_to_imu, double(0.0));
        nh_.param("filter_size_corner", fastlio.filter_size_corner_min, double(0.5));
        nh_.param("filter_size_surf", fastlio.filter_size_surf_min, double(0.5));
        nh_.param("filter_size_map", fastlio.filter_size_map_min, double(0.5));
        nh_.param("cube_side_length", fastlio.cube_len, double(200));
        nh_.param("mapping/det_range", fastlio.DET_RANGE, float(300.f));
        nh_.param("mapping/fov_degree", fastlio.fov_deg, double(180));
        nh_.param("mapping/gyr_cov", fastlio.gyr_cov, double(0.1));
        nh_.param("mapping/acc_cov", fastlio.acc_cov, double(0.1));
        nh_.param("mapping/b_gyr_cov", fastlio.b_gyr_cov, double(0.0001));
        nh_.param("mapping/b_acc_cov", fastlio.b_acc_cov, double(0.0001));
        nh_.param("mapping/publish_map", fastlio.publish_map, false);
        nh_.param("preprocess/blind", fastlio.blind, double(0.01));
        nh_.param("preprocess/lidar_type", fastlio.lidar_type, int(AVIA));
        nh_.param("preprocess/scan_line", fastlio.scan_line, int(16));
        nh_.param("preprocess/timestamp_unit", fastlio.timestamp_unit, int(US));
        nh_.param("preprocess/scan_rate", fastlio.scan_rate, int(10));
        nh_.param("point_filter_num", fastlio.point_filter_num, int(2));
        nh_.param("feature_extract_enable", fastlio.feature_extract_enable, false);
        nh_.param("runtime_pos_log_enable", fastlio.runtime_pos_log, false);
        nh_.param("mapping/extrinsic_est_en", fastlio.extrinsic_est_en, true);
        nh_.param("pcd_save/pcd_save_en", fastlio.pcd_save_en, false);
        nh_.param("pcd_save/interval", fastlio.pcd_save_interval, int(-1));
        nh_.param("mapping/extrinsic_T", fastlio.extrinT, vector<double>());
        nh_.param("mapping/extrinsic_R", fastlio.extrinR, vector<double>());

        nh_.param("/adaptive_filter/enableFilter", fastlio.enableFilter, false);
        nh_.param("/adaptive_filter/enableImu", fastlio.enableImu, false);
        nh_.param("/adaptive_filter/enableWheel", fastlio.enableWheel, false);
        nh_.param("/adaptive_filter/enableLidar", fastlio.enableLidar, false);
        nh_.param("/adaptive_filter/enableVisual", fastlio.enableVisual, false);

        nh_.param("/adaptive_filter/filterFreq", fastlio.filterFreq, std::string("l"));
        nh_.param("/adaptive_filter/freq", fastlio.freq, double(200.0));

        nh_.param("/adaptive_filter/wheelG", fastlio.wheelG, float(0.05));
        nh_.param("/adaptive_filter/imuG", fastlio.imuG, float(0.1));

        nh_.param("/adaptive_filter/alpha_lidar", fastlio.alpha_lidar, double(0.98));
        nh_.param("/adaptive_filter/alpha_visual", fastlio.alpha_visual, double(0.98));

        nh_.param("/adaptive_filter/lidarG", fastlio.lidarG, float(1000));
        nh_.param("/adaptive_filter/wheelG", fastlio.wheelG, float(0.05));
        nh_.param("/adaptive_filter/visualG", fastlio.visualG, float(0.05));
        nh_.param("/adaptive_filter/imuG", fastlio.imuG, float(0.1));

        nh_.param("/adaptive_filter/gamma_vx", fastlio.gamma_vx, float(0.05));
        nh_.param("/adaptive_filter/gamma_omegaz", fastlio.gamma_omegaz, float(0.01));
        nh_.param("/adaptive_filter/delta_vx", fastlio.delta_vx, float(0.0001));
        nh_.param("/adaptive_filter/delta_omegaz", fastlio.delta_omegaz, float(0.00001));

        nh_.param("/adaptive_filter/minIntensity", fastlio.minIntensity, double(0.0));
        nh_.param("/adaptive_filter/maxIntensity", fastlio.maxIntensity, double(1.0));

        nh_.param("/adaptive_filter/lidar_type_func", fastlio.lidar_type_func, int(2));
        nh_.param("/adaptive_filter/visual_type_func", fastlio.visual_type_func, int(2));
        nh_.param("/adaptive_filter/wheel_type_func", fastlio.wheel_type_func, int(1));

        nh_.param("/adaptive_filter/camera_type", fastlio.camera_type, int(0));
        
    }
    catch (int e)
    {
        ROS_INFO("\033[1;31mEKF-Fast-LIO2:\033[0m Exception occurred when importing parameters in Li/DARMapping Node. Exception Nr. %d", e);
    }

    // AsyncSpinner
    ros::AsyncSpinner spinner(7);
    spinner.start();

    // fast-lio intitialization
    fastlio.fastlio_initialization();

    // adaptative initialization
    fastlio.initialization_adaptive_filter();

    // ros topics
    fastlio.ros_topic_initialization();

    // adaptive filte start
    if (fastlio.enableFilter){
        ROS_INFO("\033[1;32mAdaptive Filter:\033[0m Started.");
        // runs
        fastlio.filter_start();
    }else{
        fastlio.filter_stop();
        ROS_INFO("\033[1;32mAdaptive Filter: \033[0m Stopped.");
    }

    // fast-lio run
    fastlio.run();
    
    ros::waitForShutdown();
    return 0;
}
