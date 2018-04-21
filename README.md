# Seam-Carving-OpenCV3-implemention
OpenCV3 implemention of Seam Carving for Content Aware Image Resizing in C++

Paper : http://www.win.tue.nl/~wstahw/edu/2IV05/seamcarving.pdf

Result:
![horizontal seam](https://github.com/RoyLJH/raw/master/Seam-Carving-OpenCV3-implemention/result_pics/HorizontalSeam.png)

![vertical seam](https://github.com/RoyLJH/raw/master/Seam-Carving-OpenCV3-implemention/result_pics/VerticalSeam.png)

A seam is a connected path of low energy pixels in an image. 

At each iteration, we delete one horizontal or vertical seam to reduce the size of picture (as one row or column deleted). 
Likewise we can copy one seam to expand the size of picture.

Comparisons between original pictures and processed pictures :

![1](https://github.com/RoyLJH/raw/master/Seam-Carving-OpenCV3-implemention/result_pics/1.png)

![2](https://github.com/RoyLJH/raw/master/Seam-Carving-OpenCV3-implemention/result_pics/2.png)

![3](https://github.com/RoyLJH/raw/master/Seam-Carving-OpenCV3-implemention/result_pics/3.png)

![4](https://github.com/RoyLJH/raw/master/Seam-Carving-OpenCV3-implemention/result_pics/4.png)
