///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2021, STEREOLABS.
//
// All rights reserved.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
///////////////////////////////////////////////////////////////////////////
// edited

// ----> Includes
#include <iostream>
#include <sstream>
#include <string>

#include "videocapture.hpp"

// OpenCV includes
#include <opencv2/opencv.hpp>

//#undef HAVE_OPENCV_VIZ // Uncomment if cannot use Viz3D for point cloud rendering

#ifdef HAVE_OPENCV_VIZ
#include <opencv2/viz.hpp>
#include <opencv2/viz/viz3d.hpp>
#endif

// Sample includes
#include "calibration.hpp"
#include "stopwatch.hpp"
#include "stereo.hpp"
#include "ocv_display.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
// <---- Includes

#define USE_OCV_TAPI // Comment to use "normal" cv::Mat instead of CV::UMat
//#define USE_HALF_SIZE_DISP // Comment to compute depth matching on full image frames
namespace py = pybind11;

static sl_oc::video::VideoParams make_params() {
    sl_oc::video::VideoParams p;   // calls the default ctor (HD2K, FPS_15, ERROR)
    p.res     = sl_oc::video::RESOLUTION::HD720;
    p.fps     = sl_oc::video::FPS::FPS_30;
    p.verbose = sl_oc::VERBOSITY::INFO;
    return p;
}

class VideoProcessing {
    private:
        sl_oc::VERBOSITY verbose;
        sl_oc::video::VideoParams params;
        sl_oc::video::VideoCapture cap;
        int sn;
        std::string calibration_file;
        unsigned int serial_number;
        int w,h;
        cv::Mat map_left_x, map_left_y;
        cv::Mat map_right_x, map_right_y;
        cv::Mat cameraMatrix_left, cameraMatrix_right;
        double fx, fy, cx, cy, baseline;
        cv::Mat frameBGR, left_raw, left_rect, right_raw, right_rect, frameYUV, left_for_matcher, right_for_matcher, left_disp_half,left_disp,left_disp_float, left_disp_image, left_depth_map;
        sl_oc::tools::StereoSgbmPar stereoPar;
        cv::Ptr<cv::StereoSGBM> left_matcher;
        cv::Mat cloudMat;

        #ifdef HAVE_OPENCV_VIZ
            cv::viz::Viz3d pc_viewer = cv::viz::Viz3d( "Point Cloud" );
        #endif

        uint64_t last_ts; // Used to check new frame arrival
    public:
        VideoProcessing(): cap(make_params()) {
            // ----> Set Video parameters
            params.res = sl_oc::video::RESOLUTION::HD720;
            params.fps = sl_oc::video::FPS::FPS_30;
            params.verbose = verbose;
            // <---- Set Video parameters
            
            // ----> Use Video Capture
            if( !cap.initializeVideo(-1) )
            {
                std::cerr << "Cannot open camera video capture" << std::endl;
                std::cerr << "See verbosity level for more details." << std::endl;
                throw std::runtime_error("Cannot open camera video capture");
            }
            sn = cap.getSerialNumber();
            std::cout << "Connected to camera sn: " << sn << std::endl;
            // <---- Create Video Capture

            // ZED Calibration
            serial_number = sn;
            // Download camera calibration file
            if( !sl_oc::tools::downloadCalibrationFile(serial_number, calibration_file) )
            {
                std::cerr << "Could not load calibration file from Stereolabs servers" << std::endl;
            }
            std::cout << "Calibration file found. Loading..." << std::endl;

            // ----> Frame size
            cap.getFrameSize(w,h);
            // <---- Frame size

            sl_oc::tools::initCalibration(calibration_file, cv::Size(w/2,h), map_left_x, map_left_y, map_right_x, map_right_y,
                                        cameraMatrix_left, cameraMatrix_right, &baseline);

            fx = cameraMatrix_left.at<double>(0,0);
            fy = cameraMatrix_left.at<double>(1,1);
            cx = cameraMatrix_left.at<double>(0,2);
            cy = cameraMatrix_left.at<double>(1,2);

            std::cout << " Camera Matrix L: \n" << cameraMatrix_left << std::endl << std::endl;
            std::cout << " Camera Matrix R: \n" << cameraMatrix_right << std::endl << std::endl;
            // ----> Initialize calibration

            // Note: you can use the tool 'zed_open_capture_depth_tune_stereo' to tune the parameters and save them to YAML
            if(!stereoPar.load())
            {
                stereoPar.save(); // Save default parameters.
            }

            left_matcher = cv::StereoSGBM::create(stereoPar.minDisparity,stereoPar.numDisparities,stereoPar.blockSize);
            left_matcher->setMinDisparity(stereoPar.minDisparity);
            left_matcher->setNumDisparities(stereoPar.numDisparities);
            left_matcher->setBlockSize(stereoPar.blockSize);
            left_matcher->setP1(stereoPar.P1);
            left_matcher->setP2(stereoPar.P2);
            left_matcher->setDisp12MaxDiff(stereoPar.disp12MaxDiff);
            left_matcher->setMode(stereoPar.mode);
            left_matcher->setPreFilterCap(stereoPar.preFilterCap);
            left_matcher->setUniquenessRatio(stereoPar.uniquenessRatio);
            left_matcher->setSpeckleWindowSize(stereoPar.speckleWindowSize);
            left_matcher->setSpeckleRange(stereoPar.speckleRange);

            stereoPar.print();
            // <---- Stereo matcher initialization

            #ifdef HAVE_OPENCV_VIZ
                pc_viewer = cv::viz::Viz3d( "Point Cloud" );
            #endif

            last_ts=0; // Used to check new frame arrival
            std::cout << "Done with constructor" << std::endl;
        }
        
        py::tuple grab_depth() {
            std::cout << "Starting grab depth" << std::endl;
            // Get a new frame from camera
            const sl_oc::video::Frame frame = cap.getLastFrame();
            std::cout << "Grabbed frame " << std::endl;

            // ----> If the frame is valid we can convert, rectify and display it
            if(frame.data==nullptr || frame.timestamp==last_ts) {
                int exit = -1;
                return py::make_tuple(exit, py::array_t<double>(0.0));
            }
            last_ts = frame.timestamp;

            // ----> Conversion from YUV 4:2:2 to BGR for visualization
            frameYUV = cv::Mat( frame.height, frame.width, CV_8UC2, frame.data );
            // std::cout << "Frame as Mat: " << frameYUV << std::endl;
            cv::cvtColor(frameYUV,frameBGR,cv::COLOR_YUV2BGR_YUYV);
            // <---- Conversion from YUV 4:2:2 to BGR for visualization

            // ----> Extract left and right images from side-by-side
            left_raw = frameBGR(cv::Rect(0, 0, frameBGR.cols / 2, frameBGR.rows));
            right_raw = frameBGR(cv::Rect(frameBGR.cols / 2, 0, frameBGR.cols / 2, frameBGR.rows));
            // <---- Extract left and right images from side-by-side

            // ----> Apply rectification
            sl_oc::tools::StopWatch remap_clock;
            cv::remap(left_raw, left_rect, map_left_x, map_left_y, cv::INTER_AREA );
            cv::remap(right_raw, right_rect, map_right_x, map_right_y, cv::INTER_AREA );
            double remap_elapsed = remap_clock.toc();
            std::stringstream remapElabInfo;
            remapElabInfo << "Rectif. processing: " << remap_elapsed << " sec - Freq: " << 1./remap_elapsed;
            // <---- Apply rectification

            // ----> Stereo matching
            sl_oc::tools::StopWatch stereo_clock;
            double resize_fact = 1.0;
#ifdef USE_HALF_SIZE_DISP
            resize_fact = 0.5;
            // Resize the original images to improve performances
            cv::resize(left_rect,  left_for_matcher,  cv::Size(), resize_fact, resize_fact, cv::INTER_AREA);
            cv::resize(right_rect, right_for_matcher, cv::Size(), resize_fact, resize_fact, cv::INTER_AREA);
#else
            left_for_matcher = left_rect; // No data copy
            right_for_matcher = right_rect; // No data copy
#endif


            // Apply stereo matching
            left_matcher->compute(left_for_matcher, right_for_matcher,left_disp_half);

            left_disp_half.convertTo(left_disp_float,CV_32FC1);
            //std::cout << "Left disp float: " << left_disp_float << std::endl;
            cv::multiply(left_disp_float,1./16.,left_disp_float); // Last 4 bits of SGBM disparity are decimal

#ifdef USE_HALF_SIZE_DISP
            cv::multiply(left_disp_float,2.,left_disp_float); // Last 4 bits of SGBM disparity are decimal
            //cv::UMat tmp = left_disp_float; // Required for OpenCV 3.2
            //cv::resize(tmp, left_disp_float, cv::Size(), 1./resize_fact, 1./resize_fact, cv::INTER_AREA);
#else
            left_disp = left_disp_float;
#endif
            double elapsed = stereo_clock.toc();
            std::stringstream stereoElabInfo;
            stereoElabInfo << "Stereo processing: " << elapsed << " sec - Freq: " << 1./elapsed;
            // <---- Stereo matching

            // ----> Show frames
            sl_oc::tools::showImage("Right rect.", right_rect, params.res,true, remapElabInfo.str());
            sl_oc::tools::showImage("Left rect.", left_rect, params.res,true, remapElabInfo.str());
            // <---- Show frames

            // ----> Show disparity image
            cv::add(left_disp_float,-static_cast<double>(stereoPar.minDisparity-1),left_disp_float); // Minimum disparity offset correction
            cv::multiply(left_disp_float,1./stereoPar.numDisparities,left_disp_image,255., CV_8UC1 ); // Normalization and rescaling

            cv::applyColorMap(left_disp_image,left_disp_image,cv::COLORMAP_JET); // COLORMAP_INFERNO is better, but it's only available starting from OpenCV v4.1.0

            sl_oc::tools::showImage("Disparity", left_disp_image, params.res,true, stereoElabInfo.str());
            // <---- Show disparity image

            // ----> Extract Depth map
            // The DISPARITY MAP can be now transformed in DEPTH MAP using the formula
            // depth = (f * B) / disparity
            // where 'f' is the camera focal, 'B' is the camera baseline, 'disparity' is the pixel disparity

            //std::cout << "Baseline: " << baseline << "\nfx: " << fx << std::endl;
            double num = static_cast<double>(fx*baseline);
            cv::divide(num,left_disp_float,left_depth_map);
            //std::cout << "Num: " << num << "\nLeft depth map: " << left_disp_float << std::endl;

            float central_depth = left_depth_map.at<float>(left_depth_map.rows/2, left_depth_map.cols/2 );
            std::cout << "Depth of the central pixel: " << central_depth << " mm" << std::endl;
            // <---- Extract Depth map

            // ----> Create Point Cloud
            sl_oc::tools::StopWatch pc_clock;
            size_t buf_size = static_cast<size_t>(left_depth_map.cols * left_depth_map.rows);
            std::vector<cv::Vec3d> buffer( buf_size, cv::Vec3f::all( std::numeric_limits<float>::quiet_NaN() ) );
            cv::Mat depth_map_cpu = left_depth_map;
            float* depth_vec = (float*)(&(depth_map_cpu.data[0]));
            std::cout << "Depth vec, index 0: " << *depth_vec << std::endl;

            py::array points(py::dtype::of<double>(), {left_depth_map.rows, left_depth_map.cols, 3});
            auto pts = points.mutable_unchecked<double, 3>();
            const double nan = std::numeric_limits<double>::quiet_NaN();
#pragma omp parallel for
            // Initialize return array to nan
            for (int r = 0; r < left_depth_map.rows; ++r) {
                for (int c = 0; c < left_depth_map.cols; ++c) {
                    pts(r, c, 0) = nan;
                    pts(r, c, 1) = nan;
                    pts(r, c, 2) = nan;
                }
            }

#pragma omp parallel for
            for(size_t idx=0; idx<buf_size;idx++ )
            {
                size_t r = idx/left_depth_map.cols;
                size_t c = idx%left_depth_map.cols;
                double depth = static_cast<double>(depth_vec[idx]);
                //std::cout << depth << " ";
                if(!isinf(depth) && depth >=0 && depth > stereoPar.minDepth_mm && depth < stereoPar.maxDepth_mm)
                {
                    //std::cout << "Not inf! " << depth << "   ";
                    //buffer[idx].val[2] = depth; // Z
                    //buffer[idx].val[0] = (c-cx)*depth/fx; // X
                    //buffer[idx].val[1] = (r-cy)*depth/fy; // Y
                    pts(r,c,2) = depth;
                    pts(r,c,0) = (c-cx)*depth/fx;
                    pts(r,c,1) = (r-cy)*depth/fy;
                }
            }
            return py::make_tuple(0, points);
            /*
            py_array_t<double> pts({buf_size, 3});
            auto buf = pts.request();
            double* ptr = static_cast<double*>(buf.ptr);

            for (int i = 0; i < buf_size; ++i) {
                pts[3*i] = buffer[i][0];
                pts[3*i + 1] = buffer[i][1];
                pts[3*i + 2] = buffer[i][2];
            }
            return pts;
            */
        }
};

py::array_t<double> wrap_buffer(double* ptr, size_t count) {
    return py::array_t<double>(
        {count},               // shape
        {sizeof(double)},      // stride
        ptr,                   // pointer to data
        py::none()             // no ownership (Python won't free it)
    );
}

PYBIND11_MODULE(depth_wrapper, m) {
    m.doc() = "Python interface for ZED depth";
    py::class_<VideoProcessing>(m, "VideoProcessing")
        // expose constructor (adjust argument types as needed)
        .def(py::init())
        // expose method
        .def("grab_depth", &VideoProcessing::grab_depth);
}