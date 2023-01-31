// Project 3 of CS 5330: Real-time Object 2-D Recognition
// Created by Dhruvil Parikh

// This is the training mode file. Run this file to save new objects to the database

#include "objectdetection.hpp"

int main() {

    Mat frame;
    
    // Task 1 Initialisations
    Mat src;
    Mat thresh_image;
    
    // Task 2 Initialisations
    Mat morph;

    // Task 3 Initialisations
    Mat labels;
    Mat stats;
    Mat centroids;
    Mat morph_contours;
    Mat morph_segmented;
    Mat morph_segmented_processed;
    int num_labels;
    vector<int> label_ids;

    string image_path;
    char filename[255];
    char object_name[255];
    int reset_file_flag;

    // Instructions to utilise the training mode:
    cout << "Enter the full path of the file to be added into the database: ";
    cin >> image_path;

    cout << "Enter the database file name: ";
    cin >> filename;

    cout << "Enter the label you want to be associated with this object: ";
    cin >> object_name;

    cout << "Enter '1' if you want to reset the database, else enter '0': ";
    cin >> reset_file_flag;

    frame = imread(image_path, IMREAD_COLOR);

    resize(frame, frame, Size(504, 378));
    frame.copyTo(src);

    // Task 1: Thresholding Image #######################################################################################################################################
    // thresholdImage(src, thresh_image);
    thresholdImageHSV(src, thresh_image);

    // Task 2: Morphological Operators #######################################################################################################################################
    customMorphing(thresh_image, morph);

    // Task 3: #######################################################################################################################################

    num_labels = connectedComponentsWithStats(morph, labels, stats, centroids, 4);

    morph.copyTo(morph_contours);
    morph.copyTo(morph_segmented);
    segmentedImage(morph, morph_segmented, labels, stats, label_ids, num_labels);

    grassFireGrowing(morph_segmented, morph_segmented_processed, 15);
    grassFireShrinking(morph_segmented_processed, morph_segmented_processed, 15);
    drawContoursCustom(stats, morph_contours);

    // Alternative Task 3: #############################################################################################################################
    
    vector<vector<Point>> contours;
    
    findContours(morph_segmented_processed, contours, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0, 0));
    
    vector<RotatedRect> min_rect(contours.size());
    
    for(int i = 0; i < contours.size(); i++) {
        min_rect[i] = minAreaRect(contours[i]);
    }

    // Oriented Bounding Box
    Mat morph_contours_r;
    morph_segmented.copyTo(morph_contours_r);
    cvtColor(morph_contours_r, morph_contours_r, COLOR_GRAY2BGR);
    
    for(int i = 0; i < contours.size(); i++) {
        Scalar color1 = Scalar( 255, 0, 0 );
        Scalar color2 = Scalar( 0, 0, 255 );
        // Drawing the Contour
        drawContours(morph_contours_r, contours, i, color1, 3);
        // Oriented Bounding Box
        Point2f rect_points[4];
        min_rect[i].points(rect_points);
        for (int j = 0; j < 4; j++) {
            line(morph_contours_r, rect_points[j], rect_points[(j+1)%4], color2, 3);
        }
    }

    // Task 4: Compute features for each major region #######################################################################################################################################
    
    // Calculating the Moments:
    vector<Moments> moment_image(contours.size());
    for(int i = 0; i < contours.size(); i++) {
        moment_image[i] = moments(contours[i]);
    }

    // Calculating the Centroid:
    vector<Point2f> moment_centroid(contours.size());
    for(int i = 0; i < contours.size(); i++) { 
        moment_centroid[i] = Point2f(moment_image[i].m10/moment_image[i].m00 , moment_image[i].m01/moment_image[i].m00); 
    }

    // Drawing circle where the centroids are present:
    for(int i = 0; i < contours.size(); i++) {
        Scalar color3 = Scalar(0, 255, 255);
        circle(morph_contours_r, moment_centroid[i], 4, color3, -1, 8, 0);
    }

    // Calculating the Central Axis Angle:
    vector<double> alpha(contours.size());
    Point2f first_point;
    Point2f second_point;
    for(int i = 0; i < contours.size(); i++) {
        alpha[i] = (0.5)*atan2((2*moment_image[i].mu11), (moment_image[i].mu20 - moment_image[i].mu02));
        // cout << alpha[i] << endl;
        first_point = Point2f((moment_centroid[i].x - 1000*cos(alpha[i])), (moment_centroid[i].y - 1000*sin(alpha[i])));
        second_point = Point2f((moment_centroid[i].x + 1000*cos(alpha[i])), (moment_centroid[i].y + 1000*sin(alpha[i])));
        line(morph_contours_r, first_point, second_point, Scalar(0, 255, 255), 1);
    }

    // Calculate width/length ratio, percentage filled and huMoments as the feature vectors:
    vector<double> rect_area(contours.size());
    vector<double> contour_area(contours.size());
    vector<double> percentage_filled(contours.size());
    vector<double> aspect_ratio(contours.size());
    vector<double[7]> hu_moments_image(contours.size());
    vector<vector<float>> feature_vector(contours.size());

    for(int i = 0; i < contours.size(); i++) {
        Point2f rect_points[4];
        min_rect[i].points(rect_points);
        double side1 = 0.0;
        double side2 = 0.0;

        side1 = sqrt((rect_points[0].x - rect_points[1].x)*(rect_points[0].x - rect_points[1].x) +
                (rect_points[0].y - rect_points[1].y)*(rect_points[0].y - rect_points[1].y));
        side2 = sqrt((rect_points[1].x - rect_points[2].x)*(rect_points[1].x - rect_points[2].x) +
                (rect_points[1].y - rect_points[2].y)*(rect_points[1].y - rect_points[2].y));
        
        rect_area[i] = side1*side2;
        contour_area[i] = contourArea(contours[i]);

        aspect_ratio[i] = (side1 < side2) ? (side1/side2) : (side2/side1);
        percentage_filled[i] = contour_area[i]/rect_area[i];
        HuMoments(moment_image[i], hu_moments_image[i]);

        feature_vector[i].push_back(aspect_ratio[i]);
        feature_vector[i].push_back(percentage_filled[i]);

        for(int j = 0; j < 6; j++) {
            feature_vector[i].push_back(log(abs(hu_moments_image[i][j])));
        }
        
        append_image_data_csv(filename, object_name, feature_vector[i], reset_file_flag);
    }

    return 0;
}