#include <iostream>
#include <librealsense2/rs.hpp>

int main(int argc, char* argv[]){
    rs2::pipeline p;
    p.start();

    while(1){
        rs2::frameset frames = p.wait_for_frames();
        rs2::depth_frame depth = frames.get_depth_frame();
        float width = depth.get_width();
        float height = depth.get_height();

        float dist_to_center = depth.get_distance(width/2, height/2);
        std::cout << "The camera is facing an object " << dist_to_center << "meters away\n";  
    }
    return 0;
}