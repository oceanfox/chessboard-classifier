# Chessboard Diagram Assignment Report

## Feature Extraction

The program initially represents images as a 2D array of feature vectors, flattening each image into a 1D array. For dimensionality reduction, Principal Component Analysis (PCA) is employed to reduce the image into 50 key features, before applying Linear Discriminant Analysis (LDA) to reduce further to 10. Through eigendecomposition, the 50 largest eigenvectors which we derive from the covariance matrix are used to project the mean-normalized data onto the principal component axes. Using PCA seemed to be more effective than attempting Singular Value Decomposition (though this may have been due to issues with my implementation of SVD). However, PCA's focus on eigenvectors directly aligns with capturing the most significant variations in the image data, leading to greater accuracy in feature extraction. After applying PCA, LDA is then applied to the PCA-reduced data, which has been condensed down to 50 features. The value of 50 was chosen after experimenting with different values to determine the optimal accuracy. The LDA process involves calculating the between-class scatter matrices using the means and within-class scatter matrices, which are then used to project the PCA data. The eigendecomposition in this process is represented by the equation $S_b^{-1}S_wW = vW$.  

## Square Classifier

I have used the Weighted K-Nearest Neighbour algorithm (Weighted KNN). My initial approach was using a plain KNN (K-Nearest Neighbour) algorithm however this yielded rather inconsistent results for some reason, resulting in a large amount of fluctuation (±0.5%). The algorithm calculates the euclidean distance for each feature in the samples to classify to the training samples and identifies the *k* nearest labels to each. So on top of this, the weighted KNN also gathers the distances of these labels and in my case I use its inverse as the weight. Using these weights, the nearest label is selected (with the highest weighted frequency). This label is the prediction and is appended to a list of predictions which is then returned.  

## Full-board Classification

I have again used the weighted KNN algorithm here, however this time I have implemented two chessboard logic rules, including the fact that pawns must not be in the first or the last row. As well as this, I note that although pawn promotion exists, it is not as common and have implemented a maximum value for each type of piece. If more of one class than this value in the board is found then it will be forced to re-evaluate the piece. I opted for these approaches as they were some of the few ways I could think of that could improve the results given the context of the board. To account for pawn promotion, I haven't implemented a hard limit, but more a threshold to check how dissimilar the weighted predictions are. If they pass the threshold then we assume the next most likely label. This strategy gives a slight improvement to the accuracy.

## Performance

My percentage correctness scores (to 1 decimal place) for the development data are as follows.

High quality data:

- Percentage Squares Correct: 99.4%
- Percentage Boards Correct: 99.5%

Noisy data:

- Percentage Squares Correct: 96.6%
- Percentage Boards Correct: 97.1%

## Other information

In terms of translating the images to feature vectors, I have used a gaussian filter to smooth the image and reduce noise. This seemed to improve the accuracy of the feature reduction by around 2%.

## Streamlit Demo

You can explore the classifier interactively with Streamlit:

1. Create a virtualenv and install `pip install -r requirements.txt`.
2. Run `streamlit run streamlit_app.py`.
3. Use the sidebar to select a clean/noisy model, pick a sample dev board, or upload your own 8x8 board image. The app visualises the predicted board and, when labels are available, highlights incorrect squares.
4. Open the “PCA & LDA Explorer” tab to see how squares cluster in the reduced feature spaces via interactive 3D/2D plots.

## FEN Image Generation

The `fen_image_gen.py` script synthesises 400×400 chessboards from FEN placement strings while matching the appearance of the provided training images. It first learns an average square for every (piece, square-colour) combination and then tiles those templates according to the requested FEN.

1. Build the template library from the clean training set:
   ```
   python fen_image_gen.py train --metadata data/boards.train.json --image-root data/clean --output data/piece_templates.clean.npz
   ```
2. Render a new board:
   ```
   python fen_image_gen.py render --fen "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR" --templates data/piece_templates.clean.npz --output startpos.png
   ```

The generated images are grayscale PNGs that can be fed directly into the existing preprocessing/classifier pipeline or used to augment the dataset.