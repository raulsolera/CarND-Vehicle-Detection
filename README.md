## Vehicle Detection Project

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

The code for this project can found in the Ipython Notebook `Vehicle-detection-project.ipynb` that is part of this repository.

[//]: # (Image References)
[image1]: ./report-images/car-notcar-example.png
[image2]: ./report-images/car-hog-features.png
[image3]: ./report-images/not-car-hog-features.png
[image4]: ./report-images/color-space-hog-features.png
[image5]: ./report-images/car-notcar-normalize-features.png
[image61]: ./report-images/sliding-window1.png
[image62]: ./report-images/sliding-window2.png
[image7]: ./report-images/car-detection.png
[image8]: ./report-images/detection-pipeline.png
[image9]: ./report-images/pipeline-in-sequence.png
[video1]: ./project_video.mp4


### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the second code cell of the IPython notebook in function `get_hog_features`. It uses the Opencv hog implementation (which showed to be quite slow).

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:
![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pix_per_cell` and `cells_per_block`). Random images from each of the two classes was displayed them to compare different behaviors.

Here is an example using the `RGB` color space and HOG parameters combinations of `orientations` 5 or 9, `pixels_per_cell` (8, 8) of (16, 16) and `cells_per_block` (2, 2) or (4, 4):
![alt text][image2]
![alt text][image3]

#### 2. Final choice for HOG parameters.

Performance is the main issue when using the HOG function so I decided to keep a single channel transformation (instead of getting hog features for each channel which means x3 time consumption) and watching the images there was not much difference between channels in 'RGB' space that were also similar to channel 0 in 'YCrCb' space (the image of hog transformation for gray channel was also similar to that of 'RGB' and 0 channel):
![alt text][image4]

On the other hand 'YCrCb' color space was found better option for color discrimination so this color space and channel 0 was used.

Finally an orientation of 5 and cells per block of 4 was found to have a slightly better perfomance with respecto to time, however it showed worse performance in vehicle detection hence parameters: `orientation=9`, `pix_per_cell=(8, 8)` and `cell_per_block=(2, 2)` were used for feature extraction and trining classifier.

#### 3. Classifier training.

Classification was done in four steps:
1. Feature extraction including hog features for first channel in the YCrCb color space + Histogram of colors + Spacial binning of color (both in the YCrCb color space)
2. Feature scaling to avoid some features to dominate over the others (see figure below).
3. Train and test split of the data (20% of it reserve for test)
4. Train linear SVM. This classifier was chosen as it classifies really fast but also achieves an exceptional test accuracy over 98%.
![alt text][image5]

---

### Sliding Window Search

#### 1. Sliding window search implementation.

The search space of the sliding window search was limited to the tranch between 400 and 656 pixel as it coveres all space where cars could be found.
Again the issue here is the performance of the HOG function for feature extraction so I tried to use a single HOG transformation. This implied keeping to a single window size that after some tries I found that this approach has a relevant issue as detection are limited to those cars in a certain distance. Finally I had to use two window sizes:
- (96, 96) in the tranch between 400 and 656 pixels to detect more close cars
![alt text][image61]
- (64, 64) in the tranch between 400 and 528 pixels to detect farther cars.
![alt text][image62]

The function used was that proposed by Udacity was used with minor changes to improve performance (ie. vectorizing the classification step instead of classifying each possible window, allow limited search in horizontal direction).
This, though simple, show to be robust enough to detect most cars. 

#### 2. Examples of test images to show pipeline.

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a good result. Here are some example images:
![alt text][image7]

---

### Video Implementation

#### 1.Final video output. 
Here's a [link to my video result](./result.mp4)


#### 2. Filtering results.

For each frame a car detection using the sliding windows and linear SVM was done. With all the positive windows detected a I created a heatmap with the car positions and then I applied a threshold to choose most relevant detections. I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap assuming that each one corresponds to a vehicle. Then I found the box containing each individual blob and print it on the image. The following image illustrates the pipeline:
![alt text][image8]

To improve filtering, the detection of some previous frames was used to add strength to more secure detections. A history of 6 frames including the actual was used and it resulted in a more robust pipeline. The following image show the detection pipeline in a sequence of 9 frames:
![alt text][image9]

Finally, to improve performance one every two images was processed through the detection pipeline and the detection used in the following frame. This improves processiong performances almost by a factor of 2 as the hog transformations showed to be the most time consuming process. 

#### 2. Line finding combination.

Once implemented the full vehicle detection pipeline, the funtionality of line detection from previous project was imported and combined in a single pipeline. The results can be seen in the following video:
[link to combined pipelines video result](./combined_result.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project. 

Main issues faced were the following:
1. Performance issues with HOG cv2 function. This should be exploreed further and probably a faster function than the OpenCV one might be found. Hog processing is responsible for almost 75% of the processing time of each frame.
2. Different sliding window sizes must be explored to allow for detection of farther vehicles. However this implies more HOG transformation and hence increasing performance issues.
3. Experiment with longer memory of previous frames, or assigning different weights (ie. decreasing if older) and adjusting the threshold to consider the total weight might also improve detection.
4. Finally the use of lower resolution images might also improve the performance of the processing pipeline.

