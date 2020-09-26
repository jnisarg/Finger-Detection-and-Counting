# `Finger-Detection-and-Counting`
Detects hand fingers and displays the count of the same.

This project is build on top of OpenCV.

The general working behind this project is stated below

* At start we accumulate first 60 frames to generate the background.
* Then we select roi where it can detect hand.
* Hand is detected as the largest contour of the roi which is calculated by subtracting foreground from the background.
* Once the hand is detected, we can detect the extreme points of the hand using convexhull.
* From convexhull we can obtain extreme top, bottom, left and right points which can help us to find center of the palm.
* After that a circle is drawn with the radius of ~90% of the most extreme point. This circle will seperate all fingers and wrist of the hand.
* Now fingers are counted by seprating wrist from  all the fingers.
