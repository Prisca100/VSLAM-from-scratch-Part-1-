# Visual Odometry
This is the first part of Visual SLAM. When creating a SLAM algorithm, **point clouds** are essential [see here](https://to_be_added). Point clouds are obtained using LIDARs and devices that use light or sound to determine the absolute distance between them and the object. Since cameras output images that are mainly 2D projections of the 3D world, we need a way to describe the 2D image perfectly to a system that is concerned to the 3D world such as SLAM. Thus, we identify features that are constant over multiple frames and scenarios.
Visual Odometry deals with identify these key features and estimating their motion through time and space.
We will divide this section into three main parts.

1. Feature detection and description: Obtains interest points and describes their local appearance.
2. Feature Matching and Outlier Rejection: Matching features across consecutive frames and eliminating wrongly matched features.
3. Motion Estimation: Estimating the motion of the camera through time.



## Feature Detection And Description (Detecting Interest Points)

As described in the earlier section, feature detection deals with identifying interesting points in an image. There are several interesting features that can be found in images.

1. **Edges:** This represents a part of an image where there is a rapid change in color or intensity (light)
2. **Corners:** These represent points in an image where two edges meet. Two edges mean there is a rapid change in color or brightness in two separate areas, and if they intersect, their meeting point becomes a corner.
3. **Blobs:** These represent regions or areas that are different in color or brightness and other properties compared to their neighboring pixels.
The three serve as primary features used in feature detection. T
hey are also ***keypoints*** which are basically interest points that are consistent across several images.


### Edge Detection
---
Edges represent a rapid change in color or brightness. Our human eyes can those changes fairly well but teaching that to a computer requires some way to represent these rapid changes to quantifiable ideas. Images are usually represented by an array (more accurately a tensor of numbers) with a grey scale image having just one channel (one color schema sort of) and a color image having 3 channels. For simplicity, we describe all the methods for edge detection using gray scale images. 

| ![Gray scale image matrix](../assets/images/grayscale_img_martix1.png) |
| :--: |
| *Grayscale image and image matrix (image matrix is what the computer works with)* |

Since edges are regions of rapid change, we need methods that can tell us those rapid change regions. These methods are as follows.
1. #### Gradient Based methods:
    ##### ***Gradients***
    A gradient generally represents the rate of change of a variable to another variable. For instance, when we want to find out how the distance is changing with respect to time we differentiate the distance function to get the rate of change. A similar and very popular example in the discrete domain is finding the rate of change of stock prices over given intervals. Since images are discrete functions, I shift the focus to discrete differentiation. More on continuous differentiation can be found [here](https://www.cuemath.com/calculus/differentiation/)
   ##### ***Discrete differentiation***:
    To find the rate of change of a finite function, we consider two distinct points $x_i$ and $x_{i+h}$. We also consider the function values at the two points $f(x_i)$ and $f(x_{i+h})$. To find the rate of change, we find the difference between to function values at these two points and divide it by the distance between the two points. It kinda looks like an average.
    $$
    f'(x) = \frac{f(x_{i+h})-f(x_i)}{h} 
    $$

    This is known as the forward difference method. There is also the backward difference that considers the point $x_i$ and a point before it. Additionally, we have the central difference that is the combination of the two. The formula is shown below.
    $$
    f'(x) = \frac{f(x_{i+h}) - f(x_{i-1 })}{2h}
    $$
    This is mostly used in image processing the rest of this article will focus on that. More details on deriving these equations can be found [here](https://www.dam.brown.edu/people/alcyew/handouts/numdiff.pdf)
    ##### Discrete differentiation on images:
    Digital images are usually represented in the 2D plane. This means that the function now comprises two variables. We'll call them x and y. Since there are two variables, we consider how each of them contribute to the change in the function (partial derivatives). They are as follows
    $$
    \frac{\partial f(x, y)}{\partial x} = \frac{f(x_{i+h}, y_i) - f(x_{i-h}, y_i)}{2h}
    $$
    
    $$
    \frac{\partial f(x, y)}{\partial y} = \frac{f(x_{i}, y_{i+k}) - f(x_{i}, y_{i-k})}{2k}
    $$

    assuming that h = k = 1 we end up with this:

    $$
    \frac{\partial f(x, y)}{\partial x} = \frac{f(x_{i+1}, y_i) - f(x_{i-1}, y_i)}{2}
    $$
    
    $$
    \frac{\partial f(x, y)}{\partial y} = \frac{f(x_{i}, y_{i+1}) - f(x_{i}, y_{i-1})}{2}
    $$

    The gradient for an n-variable function (in this case two) is a vector representing the partial derivatives of the functions. 
    $$
    (\frac{\partial f(x, y)}{\partial x}, \frac{\partial f(x, y)}{\partial y})
    $$
    
    To get the rate of change; the size of the edge, we find the magnitude of the gradient vector. The direction of the edge is the tan inverse of the y component over the x component. 
    Several representations of these gradient vectors include the sobel, roberts prewits, A simple convolution with both the horizontal and vertical versions of these filters give the gradient in the x and y directions respectively. More can be found [here](https://www.geeksforgeeks.org/comprehensive-guide-to-edge-detection-algorithms/)


    You can also find the implementation of these gradient based edge detectors [here](./Feature_extraction.py)

2. #### Laplacian based edge detectors
This uses the second derivative the image


### Key point detection (Blob detection)
Edges and corners are employed as features when dealing with images that have the same orientation and probably the same magnification. They however are not ideal for scenarios where images are subject to change in magnitude, orientation and intensity (brightness). This is because they are not unique enough and are repeating. To address this, some certain points in the image known as "Interest points" are identified. These interest points often called key points must have the following characteristics.
1. They must be scale invariant (magnification should have close to no effect on the features)
2. They should be orientation invariant
3. Their uniqueness must persist regardless of the brightness of the image
4. They must have a unique signature (descriptors)
5. 
6. 

Some of these features include SIFT, ORB, etc.
#### SIFT FEATURES
##### Scale Invariance
1. We take an image and we apply successive gaussian smoothing n times. Gaussian smoothing presents the illusion of viewing the image at different distances. The resulting image is characterized by the formula below.
$$I_{x,y,\sigma} = G_{x,y,\sigma} * I_{x,y}$$
2. Next we perform the difference of gaussian DoG. DoG is computed by calculating the difference between successive layers of the gaussian smoothed images. It gives the illusion of performing a Normalized Laplace of Gaussian (NLoG) on the original image. However, it allows us to fully exploit the difference in scale for different sigmas. For $n$ number of images, 
$$DoG = n-1$$
3. Next we compare each pixel in the each DoG to 26 neighbors and select local extrema as potential features
    - 9 from scale above
    - 9 form scale below
    - 8 form current scale
4. Then we repeat 1-3 for multiple octaves where the first octave starts with the original image, and the subsequent octaves downsample the image progressively usually by a factor of 2.
5. Now that we have numerous keypoints from multiple scales, we pass those keypoints through refinement algorithms that discard unqualified features. Such algorithms include Hessian matrix.

##### Orientation Invariance
---

6. Now that we have the scale invariance, we need to compute the orientation.
7. For each identified feature, a histogram is created. The histogram is made up of 36 bins (each bin storing 10 to make up $ 360\degree$)
then a circular portion proportional to the sigma of the keypoint is spread around the current keypoint.
---
the magnitude and the orientation of all the neighbors are then computed. Then the bin is constructed.

8. The orientation is stored and the magnitude adds up to the height of the bin. The orientation of the keypoint then becomes tallest bin angle. In some cases multiple orientations occur when the there are multiple bins that are greater than a certain threshold. 

##### Description Computation
---
Now that we have the keypoints, we have to find a way to describe it such that the computer can locate it in any image. Since our features are scale and orientation invariant, we can carry out the next step easily.

1. Map the feature blob until a grid of a standard size.
2. For each pixel in the blob, we compute the gradient.
3. Construct a histogram with the orientation of the gradient for each quadrant of the grid. This is because the magnitude of the gradient is not robust to changes in light intensity. 
4. Concatenate the histograms for each grid. 
5. To locate the keypoint in another image, we simply compare the sift descriptors across the images. 
We can achieve this by computing    
a. Computing the euclidean distance (l2) distance between the two histograms and declare a match where the error is below a certain threshold as shown below.
$$ d(H_1, H_2) = \sqrt{\sum(H_1(k)-H_2(k))^2} $$

or

b. Find the correlation between the two histograms and declare matches when correlation is closer to 1. See formula below
$$ r = \frac{\sum_{k=1}^n (H_1(k) - \bar{H}_1)(H_2(k) - \bar{H}_2)}{\sqrt{\sum_{k=1}^n (H_1(k) - \bar{H}_1)^2} \sqrt{\sum_{k=1}^n (H_2(k) - \bar{H}_2)^2}} $$
- **H₁(k)**: The k-th value in dataset H₁.
- **H₂(k)**: The k-th value in dataset H₂.
- **H̄₁**: The mean of H₁.
- **H̄₂**: The mean of H₂.

Check out the implementation in the "visual odometry.ipynb"

For other feature detection and description algorithms check out this [link](https://medium.com/@deepanshut041/introduction-to-orb-oriented-fast-and-rotated-brief-4220e8ec40cf). Detailed explanation coming soon.

## Feature Matching
Actually touched on this in the previous section. More details coming soon.
### Outlier Rejection

## Optical Flow Estimation
camera pose estimation, depth estimation and others coming soon. Code can be found in the visual Odometry notebook.

## Next steps (VSLAM Part 2: Mapping and Optimization)

In the next part, we will take a deeper dive into how local maps are translated to global maps and the several optimization algorithms there are for efficient loop closure. Till next time!