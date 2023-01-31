// Project 3 of CS 5330: Real-time Object 2-D Recognition
// Created by Dhruvil Parikh

// This is the main .cpp file containing the main() function.

#include "objectdetection.hpp"

int main() {

    // Capturing video using webcam
    cv::VideoCapture *cap;
    cap = new cv::VideoCapture(0);

    // Exiting if unable to open webcam
    if( !cap->isOpened() ) {
        printf("Unable to open video device\n");
        return(-1);
    }

    // Initialisations:
    Mat frame;

    bool flag_show_images = 0;
    bool flag_object_detection = 1;
    bool flag_training_mode = 0;
    bool flag_k_means = 0;

    while (true) {

        *cap >> frame;
        if( frame.empty() ) {
            printf("Frame is empty.\n");
            break;
        }

        // Calling function to completely process the image:
        objDetVideo(frame, flag_show_images, flag_object_detection, flag_training_mode, flag_k_means);
    
    }

    // Destroying all windows:
    destroyAllWindows();

    // Freeing up the memory:
    delete cap;
    return 0;

}