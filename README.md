# Shape Identification and Beautification Project

## Overview

The Shape Identification and Beautification Project is an advanced initiative designed to identify, regularize, and enhance geometric shapes within images. By leveraging a combination of image processing techniques and machine learning algorithms, the project aims to produce accurate, aesthetically pleasing representations of geometric shapes. Key methodologies include uniform sampling, Principal Component Analysis (PCA) alignment, Hungarian matching, and symmetry analysis.

## Tech Stack

Our project utilizes a robust set of tools and libraries:

- **OpenCV**: 
  - *Purpose*: Facilitates image preprocessing, contour detection, and shape approximation.
  - *Key Functions*: `cv2.imread()`, `cv2.cvtColor()`, `cv2.GaussianBlur()`, `cv2.Canny()`, `cv2.findContours()`, `cv2.drawContours()`, `cv2.approxPolyDP()`
  
- **NumPy**: 
  - *Purpose*: Provides numerical operations and array manipulation capabilities.
  - *Key Functions*: `np.array()`, `np.linspace()`, `np.sqrt()`, `np.sum()`
  
- **SciPy**: 
  - *Purpose*: Used for mathematical and optimization operations, including distance metrics.
  - *Key Functions*: `scipy.spatial.distance`, `scipy.optimize.linear_sum_assignment()`
  
- **Matplotlib**: 
  - *Purpose*: Assists in visualizing results and outputs.
  - *Key Functions*: `plt.imshow()`, `plt.title()`, `plt.axis()`
  
- **Shapely**: 
  - *Purpose*: Supports geometric operations and shape analysis (potential for future extensions).
  - *Key Functions*: Not directly used in the current implementation but useful for future extensions.
  
- **Scikit-learn**: 
  - *Purpose*: Provides machine learning tools for PCA-based shape alignment.
  - *Key Functions*: `sklearn.decomposition.PCA`

## Installation

To set up the project environment, ensure you have Python installed and execute the following commands to install the required libraries:

```bash
pip install opencv-python numpy scipy matplotlib scikit-learn
```

For environments where GUI features of OpenCV are not needed, use the headless version:

```bash
pip install opencv-python-headless
```

## How It Works

### 1. Image Preprocessing

**Grayscale Conversion**: 
- Converts the image to grayscale to simplify processing.
- *Implementation*: `cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)`
  
**Blurring**: 
- Applies Gaussian blur to smooth the image and reduce noise.
- *Implementation*: `cv2.GaussianBlur()` with a (5, 5) kernel.
  
**Edge Detection**: 
- Uses the Canny edge detection algorithm to identify edges and contours.
- *Implementation*: `cv2.Canny()` function to detect edges and produce a binary image.

### 2. Contour Extraction

**Find Contours**: 
- Extracts contours from the edge-detected image, representing the boundaries of shapes.
- *Implementation*: `cv2.findContours()` retrieves contours from the binary edge image.

### 3. Shape Processing

**Uniform Sampling**:
- Samples points along each contour uniformly for consistent shape representation.
- *Implementation*: `uniform_sampling()` function calculates the arc length and samples points using `np.linspace()`.

**Shape Approximation**:
- Simplifies contours into polygons for easier analysis.
- *Implementation*: `cv2.approxPolyDP()` approximates contours to polygons based on specified accuracy.

### 4. Shape Analysis and Regularization

**PCA Alignment**:
- Aligns shapes using Principal Component Analysis (PCA) for standardized orientation.
- *Implementation*: `sklearn.decomposition.PCA` aligns shapes along their major axis.

**Hungarian Matching**:
- Matches contours with reference shapes using the Hungarian algorithm to find the best fit.
- *Implementation*: `scipy.optimize.linear_sum_assignment()` minimizes the cost matrix for optimal shape matching.

**Symmetry Analysis**:
- Enhances shapes by ensuring symmetry, improving their aesthetic quality.
- *Implementation*: Analyzes and adjusts shapes to align with symmetric reference shapes, enhancing visual appeal.

### 5. Counting Shapes

**Count Shapes**:
- Determines the number of shapes detected and processed in the image.
- *Implementation*: Counts contours and classifies them based on their approximation.

### 6. Regularization

**Regularize Shapes**:
- Applies techniques to ensure shapes are uniform and consistent with reference models.
- *Implementation*: Shapes are adjusted using uniform sampling, PCA alignment, and Hungarian matching for consistency and visual appeal.

## Algorithms

### Uniform Sampling

1. **Calculate Arc Length**: Compute the arc length of the contour using `cv2.arcLength()`.
2. **Sample Points**: Use `np.linspace()` to evenly sample points along the contour based on the arc length.
3. **Interpolate Points**: Interpolate points along the contour to ensure uniform distribution.

### PCA Alignment

1. **Center Data**: Center the data by subtracting the mean.
2. **Compute Covariance Matrix**: Calculate the covariance matrix of the centered data.
3. **Perform PCA**: Use `sklearn.decomposition.PCA` to perform PCA and obtain principal components.
4. **Align Shapes**: Transform shapes to align them with the principal components.

### Hungarian Matching

1. **Create Cost Matrix**: Compute the cost matrix representing differences between detected shapes and reference shapes.
2. **Solve Assignment Problem**: Use `scipy.optimize.linear_sum_assignment()` to solve the assignment problem and find the optimal matching.
3. **Transform Shapes**: Apply the transformation to align shapes with the closest reference shapes based on the optimal assignment.

### Symmetry Analysis

1. **Analyze Symmetry**: Check for symmetry properties in shapes.
2. **Transform Shapes**: Adjust shapes to ensure they match symmetric reference shapes.
3. **Enhance Visual Appeal**: Make geometric adjustments to improve the visual balance and appeal of shapes.

## How to Run the Project

1. **Setup Environment**:
   - Ensure Python is installed.
   - Install required packages using pip:

```bash
pip install opencv-python numpy scipy matplotlib scikit-learn
```

2. **Prepare Images**:
   - Place the images you want to process in the project directory.

3. **Run the Script**:
   - Execute the Python script to process images:

```bash
python your_script_name.py
```

   - The script will process images, identify and regularize shapes, and save the results with `regularized_` prefixed to the original filenames.

4. **View Results**:
   - Processed images will be saved in the same directory with enhanced shapes.
   - Results are also displayed using matplotlib for visual confirmation.

## Future Scope

1. **Enhanced Shape Recognition**:
   - Integrate machine learning techniques to recognize more complex shapes and patterns.

2. **Real-Time Processing**:
   - Develop algorithms for real-time shape identification and beautification in video streams.

3. **Extended Shape Library**:
   - Expand the reference shape library to include more diverse geometries.

4. **User Interface**:
   - Create an interactive interface for shape analysis and processing.

5. **Integration with Augmented Reality (AR)**:
   - Combine shape recognition with AR technologies for real-time shape enhancement in AR applications.

## Contributions

- **Anurag Singh Thakur**: Captain
- **Aarif Khan**
- **Pragya Dave**

## References

- **OpenCV Documentation**: [OpenCV](https://docs.opencv.org/)
- **NumPy Documentation**: [NumPy](https://numpy.org/doc/)
- **SciPy Documentation**: [SciPy](https://docs.scipy.org/doc/scipy/)
- **Matplotlib Documentation**: [Matplotlib](https://matplotlib.org/stable/contents.html)
- **Scikit-learn Documentation**: [Scikit-learn](https://scikit-learn.org/stable/)

---
