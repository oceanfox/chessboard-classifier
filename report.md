# Chessboard Diagram Assignment Report

[Replace the square-bracketed text with your own text. *Leave everything else unchanged.*
Note, the reports are parsed to check word limits, etc. Changing the format may cause
the parsing to fail.]

## Feature Extraction (Max 200 Words)

The program initially represents images as a 2D array of feature vectors, flattening each image into a 1D array. For dimensionality reduction, Principal Component Analysis (PCA) is employed to reduce the image into 10 key features. Through eigendecomposition, the 10 largest eigenvectors which we derive from the covariance matrix are used to project the mean-normalized data onto the principal component axes. Using PCA seemed to be more effective than attempting Singular Value Decomposition (though this may have been issues with my implementation of SVD), however PCA's focus on eigenvectors directly aligns with capturing the most significant variations in the image data, leading to superior accuracy in feature reduction.

## Square Classifier (Max 200 Words)

[Describe and justify the design of your classifier and any associated classifier training
stage.]

## Full-board Classification (Max 200 Words)



## Performance

My percentage correctness scores (to 1 decimal place) for the development data are as follows.

High quality data:

- Percentage Squares Correct: [Insert percentage here, e.g. 98.1% - remove brackets]
- Percentage Boards Correct: [Insert percentage here, e.g. 98.0% - remove brackets]

Noisy data:

- Percentage Squares Correct: [Insert percentage here, e.g. 83.3% - remove brackets]
- Percentage Boards Correct: [Insert percentage here, e.g. 58.1% - remove brackets]

## Other information (Optional, Max 100 words)

[Optional: highlight any significant aspects of your system that are NOT covered in the
sections above]
