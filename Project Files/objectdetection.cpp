// Project 3 of CS 5330: Real-time Object 2-D Recognition
// Created by Dhruvil Parikh

#include <iostream>

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <math.h>

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include "csv_util.hpp"

using namespace std;
using namespace cv;

bool sortPairedVector(const pair<string, float> &a, const pair<string, float> &b)
{
    return (a.second < b.second);
}

void thresholdImage(const Mat &src, Mat &thresh_image) {

    src.copyTo(thresh_image);
    cvtColor(thresh_image, thresh_image, COLOR_BGR2GRAY);

    int threshold = 64;
    int i = 0, j = 0, c = 0;

    for(i = 0; i < src.rows; i++) {
        for(j = 0; j < src.cols; j++) {
            
            thresh_image.at<unsigned char>(i, j) = ((src.at<Vec3b>(i, j)[0] > threshold && src.at<Vec3b>(i, j)[1] > threshold && src.at<Vec3b>(i, j)[2] > threshold) ? 0 : 255);

        }
    }

}

void thresholdImageHSV(const Mat &src_original, Mat &thresh_image) {

    Mat src;
    src_original.copyTo(thresh_image);
    cvtColor(src_original, src, COLOR_BGR2HSV);
    cvtColor(thresh_image, thresh_image, COLOR_BGR2GRAY);

    int saturation = 55;
    int value = 80;

    // int saturation = 85;
    // int value = 130;

    // int saturation = 125;
    // int value = 70;
    
    int i = 0, j = 0, c = 0;

    for(i = 0; i < src.rows; i++) {
        for(j = 0; j < src.cols; j++) {
            
            thresh_image.at<unsigned char>(i, j) = ((src.at<Vec3b>(i, j)[1] < saturation && src.at<Vec3b>(i, j)[2] > value) ? 0 : 255);

        }
    }

}

void inbuiltMorph(const Mat &thresh_image, Mat &morph) {

    Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
    erode(thresh_image, morph, kernel, Point(-1, -1), 1);
    dilate(morph, morph, kernel, Point(-1, -1), 1);
    erode(morph, morph, kernel, Point(-1, -1), 2);
    dilate(morph, morph, kernel, Point(-1, -1), 2);

}

void grassFireGrowing(const Mat &image, Mat &morph, int cnt) {

    Mat distMat(image.rows, image.cols, CV_16S);
    image.copyTo(morph);
    int i, j;

    for(i = 0; i < distMat.rows; i++) {
        for(j = 0; j < distMat.cols; j++) {
            if(image.at<unsigned char>(i, j) == 255) {
                distMat.at<unsigned short>(i, j) = 0;
            } else {
                distMat.at<unsigned short>(i, j) = 32767;
            }
        }
    }

    for(i = 0; i < distMat.rows; i++) {
        for(j = 0; j < distMat.cols; j++ ) {

            if (distMat.at<unsigned short>(i, j) != 0) {

                if (i==0 || j==0) {
                    distMat.at<unsigned short>(i, j) = 32767;
                } else {
                    distMat.at<unsigned short>(i, j) = min(distMat.at<unsigned short>(i - 1, j), distMat.at<unsigned short>(i, j - 1)) + 1;
                }
            }
        }
    }

    for(i = distMat.rows - 2; i >= 0; i--) {
        for(j = distMat.cols - 2; j >= 0; j--) {

            if (distMat.at<unsigned short>(i, j) != 0) {
                distMat.at<unsigned short>(i, j) = (distMat.at<unsigned short>(i, j) < min(distMat.at<unsigned short>(i + 1, j), distMat.at<unsigned short>(i, j + 1)) + 1) 
                                                ? distMat.at<unsigned short>(i, j) : min(distMat.at<unsigned short>(i + 1, j), distMat.at<unsigned short>(i, j + 1)) + 1;
            }
        }
    }

    for (i = 0; i < morph.rows; i++) {
        for (j = 0; j < morph.cols; j++) {

            if(distMat.at<unsigned short>(i, j) <= cnt) {
                morph.at<unsigned char>(i, j) = 255;
            }

        }
    }

}

void grassFireShrinking(const Mat &image, Mat &morph, int cnt) {

    Mat distMat;
    image.copyTo(distMat);
    image.copyTo(morph);
    int i, j;

    for(i = 0; i < distMat.rows; i++) {
        for(j = 0; j < distMat.cols; j++ ) {

            if (distMat.at<unsigned char>(i, j) != 0) {

                if (i==0 || j==0) {
                    // distMat.at<unsigned char>(i, j) = 1;
                } else {
                    distMat.at<unsigned char>(i, j) = min(distMat.at<unsigned char>(i - 1, j), distMat.at<unsigned char>(i, j - 1)) + 1;
                }
            }
        }
    }

    for(i = distMat.rows - 2; i >= 0; i--) {
        for(j = distMat.cols - 2; j >= 0; j--) {

            if (distMat.at<unsigned char>(i, j) != 0) {

                distMat.at<unsigned char>(i, j) = (distMat.at<unsigned char>(i, j) < min(distMat.at<unsigned char>(i + 1, j), distMat.at<unsigned char>(i, j + 1)) + 1) 
                                                ? distMat.at<unsigned char>(i, j) : min(distMat.at<unsigned char>(i + 1, j), distMat.at<unsigned char>(i, j + 1)) + 1;
            }
        }
    }

    for (i = 0; i < morph.rows; i++) {
        for (j = 0; j < morph.cols; j++) {

            if(distMat.at<unsigned char>(i, j) <= cnt) {
                morph.at<unsigned char>(i, j) = 0;
            }

        }
    }

}

void customMorphing(const Mat &thresh_image, Mat &morph) {

    // grassFireShrinking(thresh_image, morph, 3);
    // grassFireGrowing(morph, morph, 3);
    // grassFireShrinking(morph, morph, 5);
    // grassFireGrowing(morph, morph, 5);

    grassFireShrinking(thresh_image, morph, 1);
    grassFireGrowing(morph, morph, 9);
    grassFireShrinking(morph, morph, 11);
    grassFireGrowing(morph, morph, 3);

    // grassFireGrowing(thresh_image, morph, 9);
    // grassFireShrinking(morph, morph, 6);
    // grassFireGrowing(morph, morph, 3);
    // grassFireShrinking(morph, morph, 6);

}

void drawContoursCustom(const Mat &stats, Mat &morph_contours) {

    cvtColor(morph_contours, morph_contours, COLOR_GRAY2BGR);

    for(int i=0; i<stats.rows; i++) {
        int x = stats.at<int>(Point(0, i));
        int y = stats.at<int>(Point(1, i));
        int w = stats.at<int>(Point(2, i));
        int h = stats.at<int>(Point(3, i));
        int area = stats.at<int>(Point(4, i));

        Scalar color(0, 255, 0);
        Rect rect(x,y,w,h);
        rectangle(morph_contours, rect, color, 3);
    }

}

void segmentedImage(const Mat &morph, Mat &morph_segmented, Mat &labels, Mat &stats, vector<int> &label_ids, int num_labels) {

    double area;

    for(int i = 1; i < num_labels; i++) {
        if(stats.at<int>(i, CC_STAT_AREA) > 500) { //950) {
            label_ids.push_back(i);
        }
    }

    for(int i = 0; i < labels.rows; i++) {
        for(int j = 0; j < labels.cols; j++) {
            if(i == 0 || i == (labels.rows - 1) || j == 0 || j == (labels.cols - 1)) {
                label_ids.erase(remove(label_ids.begin(), label_ids.end(), labels.at<int>(i, j)), label_ids.end());
            }
        }
    }

    for(int i = 0; i < morph.rows; i++) {
        for(int j = 0; j < morph.cols; j++) {

            if(find(label_ids.begin(), label_ids.end(), labels.at<int>(i, j)) != label_ids.end()) {
                morph_segmented.at<unsigned char>(i, j) = 255;
            } else {
                morph_segmented.at<unsigned char>(i, j) = 0;
            }
        
        }
    }

}

void standardDeviation(vector<vector<float>> &retrieved_feature_vector, vector<float> &std_dev) {

    vector<float> mean(retrieved_feature_vector[0].size());

    for(int j = 0; j < std_dev.size(); j++) {

        vector<float> temp_column_input(retrieved_feature_vector.size());
        Mat mean_temp, std_dev_temp;
        int cnt = 0;
        for(int i = 0; i < retrieved_feature_vector.size(); i++) {

            temp_column_input[i] = retrieved_feature_vector[i][j];

        }
        meanStdDev(temp_column_input, mean_temp, std_dev_temp);
        std_dev[j] = std_dev_temp.at<double>(0);

    }

}

float scaledEuclideanDistance(vector<float> feature_vector_c, vector<float> retrieved_feature_vector_i, const vector<float> std_dev ) {

    float diff = 0;
    float squared_normalised_diff = 0;
    float sum = 0;
    float scaled_euclidean_distance = 0;

    for(int j = 0; j < retrieved_feature_vector_i.size(); j++) {

        diff = feature_vector_c[j] - retrieved_feature_vector_i[j];
        squared_normalised_diff = (diff/std_dev[j])*(diff/std_dev[j]);
        sum += squared_normalised_diff;
        
    }

    scaled_euclidean_distance = sum/retrieved_feature_vector_i.size();

    return scaled_euclidean_distance;

}

void kmeans(vector<vector<float>> &feature_vector, vector<vector<Point>> &contours) {

    char filename[255] = "kmeans.csv";
    vector<vector<float>> retrieved_feature_vector;
    vector<char *> retrieved_object_names;

    read_image_data_csv(filename, retrieved_object_names, retrieved_feature_vector);

    int k = 3;

    vector<vector<float>> k_centers(k);
    vector<vector<float>> k_centers_prev(k);

    for(int i = 0; i < k; i++) {
        k_centers[i] = retrieved_feature_vector[i + 5];
    }

    vector<float> std_dev(retrieved_feature_vector[0].size());
    standardDeviation(retrieved_feature_vector, std_dev);

    vector<float> labels(retrieved_feature_vector.size());

    vector<string> label_object(retrieved_object_names.size());

    int epochs = 15;
    int flag_break = 0;

    for(int iter = 0; iter < epochs; iter++) {

        for(int i = 0; i < retrieved_feature_vector.size(); i++) {

            float scaled_euclidean_distance_0 = 0;
            float scaled_euclidean_distance_1 = 0;
            float scaled_euclidean_distance_2 = 0;
            float min_distance = 0;

            scaled_euclidean_distance_0 = scaledEuclideanDistance(k_centers[0], retrieved_feature_vector[i], std_dev);
            scaled_euclidean_distance_1 = scaledEuclideanDistance(k_centers[1], retrieved_feature_vector[i], std_dev);
            scaled_euclidean_distance_2 = scaledEuclideanDistance(k_centers[2], retrieved_feature_vector[i], std_dev);

            min_distance = scaled_euclidean_distance_0 < scaled_euclidean_distance_1 ? scaled_euclidean_distance_0 : scaled_euclidean_distance_1;
            min_distance = min_distance < scaled_euclidean_distance_2 ? min_distance : scaled_euclidean_distance_2;

            if(min_distance == scaled_euclidean_distance_0) {
                labels[i] = 0;
            }

            if(min_distance == scaled_euclidean_distance_1) {
                labels[i] = 1;
            }

            if(min_distance == scaled_euclidean_distance_2) {
                labels[i] = 2;
            }

        }

        k_centers_prev = k_centers;
        float num_features_0 = 1, num_features_1 = 1, num_features_2 = 1;

        for(int i = 0; i < retrieved_feature_vector.size(); i++) {

            if(labels[i] == 0) {

                for(int j = 0; j < retrieved_feature_vector[0].size(); j++) {
                    k_centers[0][j] += retrieved_feature_vector[i][j];
                }
                num_features_0++;
            }

            if(labels[i] == 1) {

                for(int j = 0; j < retrieved_feature_vector[0].size(); j++) {
                    k_centers[1][j] += retrieved_feature_vector[i][j];
                }
                num_features_1++;
            }

            if(labels[i] == 2) {

                for(int j = 0; j < retrieved_feature_vector[0].size(); j++) {
                    k_centers[2][j] += retrieved_feature_vector[i][j];
                }
                num_features_2++;
            }

        }

        for(int j = 0; j < k_centers[0].size(); j++) {

            k_centers[0][j] = k_centers[0][j]/num_features_0;

            k_centers[1][j] = k_centers[1][j]/num_features_1;

            k_centers[2][j] = k_centers[2][j]/num_features_2;

            if (k_centers == k_centers_prev) {
                // cout << "Algorithm converged at the " << iter << "th iteration." << endl;
                flag_break = 1;
                break;
            }

        }

        if(flag_break) {
            break;
        }

    }

    vector<float> std_dev_centers(k_centers[0].size());
    standardDeviation(k_centers, std_dev_centers);

    for(int c = 0; c < contours.size(); c++) {

        float scaled_euclidean_distance_0 = 0;
        float scaled_euclidean_distance_1 = 0;
        float scaled_euclidean_distance_2 = 0;
        float min_distance = 0;

        scaled_euclidean_distance_0 = scaledEuclideanDistance(k_centers[0], feature_vector[c], std_dev_centers);
        scaled_euclidean_distance_1 = scaledEuclideanDistance(k_centers[1], feature_vector[c], std_dev_centers);
        scaled_euclidean_distance_2 = scaledEuclideanDistance(k_centers[2], feature_vector[c], std_dev_centers);

        min_distance = scaled_euclidean_distance_0 < scaled_euclidean_distance_1 ? scaled_euclidean_distance_0 : scaled_euclidean_distance_1;
        min_distance = min_distance < scaled_euclidean_distance_2 ? min_distance : scaled_euclidean_distance_2;
        
        if(min_distance == scaled_euclidean_distance_0) {
            label_object[c] = "Trimmer";
        }

        if(min_distance == scaled_euclidean_distance_1) {
            label_object[c] = "Beats Ear Buds Case";
        }

        if(min_distance == scaled_euclidean_distance_2) {
            label_object[c] = "Spatula";
        }
        
        cout << "Matching Object as per K-means Clustering: " << label_object[c] << endl;
        cout << "K-means Distance from the nearest centroid: " << min_distance << endl;

    }

    // KNN Classification:
    int knn = 2;

    for(int c = 0; c < contours.size(); c++) {

        vector<pair<string, float>> distance_metric;
        
        for(int i = 0; i < retrieved_feature_vector.size(); i++) {

            float scaled_euclidean_distance = 0;
            scaled_euclidean_distance = scaledEuclideanDistance(feature_vector[c], retrieved_feature_vector[i], std_dev);

            distance_metric.push_back({retrieved_object_names[i], scaled_euclidean_distance});

        }

        sort(distance_metric.begin(), distance_metric.end(), sortPairedVector);

        cout << "Object Classification with KNN: " << distance_metric[0].first << endl;
        cout << "Distance from Training Data: 1st Nearest = " << distance_metric[0].second << " 2nd Nearest = " << distance_metric[1].second << endl;
     
    }

}

void objDetVideo(const Mat &frame, bool flag_show_images, bool flag_object_detection, bool flag_training_mode, bool flag_k_means) {
    
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

    char filename[255] = "data.csv";
    int reset_file_flag = 0;

    resize(frame, src, Size(504, 378));

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
        
    }

    // Starting initializations for Object Detection:
    vector<vector<float>> retrieved_feature_vector;
    vector<char *> retrieved_object_names;

    read_image_data_csv(filename, retrieved_object_names, retrieved_feature_vector);
    // system("clear");

    vector<float> std_dev(retrieved_feature_vector[0].size());

    standardDeviation(retrieved_feature_vector, std_dev);

    for(int c = 0; c < contours.size(); c++) {

        vector<pair<string, float>> distance_metric;
        
        for(int i = 0; i < retrieved_feature_vector.size(); i++) {

            float scaled_euclidean_distance = 0;
            scaled_euclidean_distance = scaledEuclideanDistance(feature_vector[c], retrieved_feature_vector[i], std_dev);

            distance_metric.push_back({retrieved_object_names[i], scaled_euclidean_distance});

        }

        sort(distance_metric.begin(), distance_metric.end(), sortPairedVector);

        cout << "Object Name: " << distance_metric[0].first << endl;
        cout << "Distance from Training Data: " << distance_metric[0].second << endl;
        putText(morph_contours_r, distance_metric[0].first, Point(moment_centroid[c]), FONT_HERSHEY_DUPLEX, 0.6, Scalar(255, 0, 255), 2);  

    }

    // Displaying Images

    imshow("Original", src);
    imshow("Thresholded", thresh_image);
    imshow("Morphed", morph);
    imshow("Contours", morph_contours);
    imshow("Segmented", morph_segmented_processed);
    imshow("Rotated Bounding Box", morph_contours_r);
    waitKey();

    // Predicting the class using k-means:    
    kmeans(feature_vector, contours);
    
}