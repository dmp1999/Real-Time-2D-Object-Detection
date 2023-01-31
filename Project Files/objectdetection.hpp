// Project 3 of CS 5330: Real-time Object 2-D Recognition
// Created by Dhruvil Parikh

// This is the main header file that has all the required functions.

#include "objectdetection.cpp"

// Custom function to sort vector on the second element:
bool sortPairedVector(const pair<string, float> &a, const pair<string, float> &b);

// Custom function to threshold image on BGR values:
void thresholdImage(const Mat &src, Mat &thresh_image);

// Custom function to threshold image on saturation and value in HSV color space:
void thresholdImageHSV(const Mat &src_original, Mat &thresh_image);

// Custom Function to implement in-built morph functions:
void inbuiltMorph(const Mat &thresh_image, Mat &morph);

// Custom Function to implement user-defined grassfire transform and growing:
void grassFireGrowing(const Mat &image, Mat &morph, int cnt);

// Custom Function to implement user-defined grassfire transform and shringking:
void grassFireShrinking(const Mat &image, Mat &morph, int cnt);

// Custom function to integrate grassfire growing and shrinking functions:
void customMorphing(const Mat &thresh_image, Mat &morph);

// Custom function to draw contours:
void drawContoursCustom(const Mat &stats, Mat &morph_contours);

// Custom function for region segmentation:
void segmentedImage(const Mat &morph, Mat &morph_segmented, Mat &labels, Mat &stats, vector<int> &label_ids, int num_labels);

// Custom function to calculate standard deviation of columns in a given database:
void standardDeviation(vector<vector<float>> &retrieved_feature_vector, vector<float> &std_dev);

// Custom function to calculate scaled euclidean distance given two feature vectors:
float scaledEuclideanDistance(vector<float> feature_vector_c, vector<float> retrieved_feature_vector_i, const vector<float> std_dev);

// Custom function to implement kmeans clustering as well as knn classifier:
void kmeans(vector<vector<float>> &feature_vector, vector<vector<Point>> &contours);

// Custom function for complete processing of the image:
void objDetVideo(const Mat &frame, bool flag_show_images, bool flag_object_detection, bool flag_training_mode, bool flag_k_means);