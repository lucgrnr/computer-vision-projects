# 2.1.1 Handcrafted features
Reflect on the parameters you’ve chosen. How does this feature compare to your previous grabbing task in the individual assignment?
Did you need specific pre-processing steps before computing these feature descriptors on your images (which ones and why)? Did the visualisation show good discriminative and robustness properties?

* Parameters: Explain that we ran multiple tests
* Comparison with grabbing:
* Visualisation: t-SNE shows excellent separation (two round areas) 

# 2.1.2 Learning features from data: PCA
Reflect on your choice of 𝑝 (optimal number of principal components) and how this might have influenced the performance of the features that you learned. Did you need specific pre-processing steps before computing these feature descriptors on your images (which ones and why)? How many non-zero eigenvalues did you have, why is this? Did the visualisation show good discriminative and robustness properties?

* Optimal p:
* Pre-processing:
* Number of eigen-values
* Visualisation:

# 2.2 Classification
How did you build this classifier for each of your feature representations? Do the models hold up when you predict on images of Michael Cera and Sarah Hyland?

Started with simple LinearSVM
Then benchmarked classifiers: explain results
Good performance for separating Mila and Jesse, but still some errors with Jesse/Michael and Mila/Sarah

# 2.3 Improve performance
Clearly document your zero-to-hero journey in the submitted notebook: Which iterations did you go through? Do you notice improvements?

We started working in parallel on face detection, HOG and PCA.
Our initial HOG and PCA were performed on the full images, before faces were detected. The performance was not bad, significantly better than the random classifier.
For face detection, we implemented two variants: HAAR and DNN:
* HAAR:
* DNN:
Once we were happy with the face detection algorithm, we passed the cropped faces into HOG and PCA which increased performance.
We proceeded to improve the perform of HOG and PCA by finetuning using cross-validation (with a simple classifier such as RBF SVM).
In parallel, we also worked on augmenting the training: our previous face detection algorithm only extracted one face per image. But in some cases, there are multiple faces. To augment the data, we did the following:
* Extract all faces: when multiple faces, extract all of them and add them to dataset. When no faces, also add as class 0.
* Duplicate images with slightly rotated versions, vertically flipped and updated lighting.

The top classifiers were achieving around 80% accuracy for HOG and 70% for PCA on the cross-validation set.
To further increase, accuracy we turned our attention to neural networks. Given that we have very few training data, we don't train a neural network from scratch and rely instead on transfer learning by using MobileNetV2.

For our final implementation, we implement significant changes:
* Switch to 5 classes: 
* FaceNet neural network: Our initial MobileNetV2 neural is very fast, but general purpose. It is trained to recognise many different objects. 

# 2.4 Discussion
Summarize (very briefly) what you have learned in this assignment. Discuss qualitative differences between the different feature representations that you constructed. Why/when would one work better than the other? Why is performance (sub) optimal? What would you do better/more if you would had plenty of time to spend on each step?

* Learned:
* HOG vs PCA: 
* Further improvement: collect more balanced training data. Our current data is unbalanced: almost no people other than Jesse, Mila, Michael and Sarah. This makes it more difficult for our HOG and PCA classifiers to identify when pictures show someone else.