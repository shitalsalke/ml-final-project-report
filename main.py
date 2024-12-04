import streamlit as st

# Title and Team Information
st.title('CS7641: Project Midterm Report')

st.subheader('(Group 42)')

st.header('Tumor Detection and Classification')
st.markdown('**Team Members:** Shital Salke, Jenny Lin, Koushika Kesavan, Hima Varshini Parasa')
st.markdown('**Institution:** Georgia Institute of Technology')

# Introduction and Background Section

st.header('1. Introduction and Background')
st.write("""
Brain tumors are abnormal cell growths in or around the brain, classified as benign (non-cancerous) or malignant (cancerous). In adults, the primary types include Gliomas (usually malignant), Meningiomas (often benign), and Pituitary tumors (generally benign). Early diagnosis is crucial for improving outcomes and treatment options. This project aims to develop a machine learning model for detecting and classifying brain tumors using MRI images.

Several studies highlight the effectiveness of modern techniques. A YOLOv7-based model with attention mechanisms achieved 99.5% accuracy [1]. Transfer learning models like VGG16 reached 98% accuracy, outperforming traditional CNNs [2]. A CNN optimized with the Grey Wolf Optimization algorithm achieved 99.98% accuracy [3], while MobileNetv3 reached 99.75%, surpassing ResNet and DenseNet [4]. Additionally, one study compared CNNs (96.47%) with Random Forest (86%) for tumor classification [5].

The dataset for this project, “Brain Tumor (MRI scans),” sourced from Kaggle, contains 3,264 MRI images across three tumor types: Gliomas, Meningiomas, and Pituitary tumors, with a balanced distribution of images in various orientations.
""")
st.markdown('Link to the Dataset: [https://www.kaggle.com/datasets/rm1000/brain-tumor-mri-scans/data](https://www.kaggle.com/datasets/rm1000/brain-tumor-mri-scans/data)')

# Problem Definition Section

st.header('2. Problem Definition')
st.write("""
Manual analysis of MRI scans by radiologists is labor-intensive and error-prone. This project aims to automate the detection and classification of brain tumors (Gliomas, Meningiomas, and Pituitary tumors) to improve accuracy and efficiency compared to existing models.

The solution involves developing a deep learning model using a custom CNN with multiple convolutional, pooling, and fully connected layers. Optimization will utilize Transfer Learning with models like VGG16, ResNet50, and EfficientNet. Performance will be assessed using metrics such as Precision, Recall, Accuracy, F1 Score, and ROC-AUC.

**Difference From Prior Literature**: Our approach enhances existing methods by utilizing EfficientNetB2 for effective feature extraction and incorporating unsupervised techniques like DBSCAN and GMM to address noise and irregular tumor shapes. This combination leads to improved model performance and faster, more accurate tumor classification.
""")

# Goals section

st.header('3. Project Goals')
st.write("""
- Detects and classifies brain tumors through MRI scans.
- Reduce false positives and detect tumors early.
- Generates consistent accuracies for different types of brain tumors.
- Generalizes well to unseen MRI scans.
""")

# Methods Section

st.header("4. Methods")

st.subheader("Data Preprocessing Methods")
st.write("""
To prepare MRI images for effective model training, preprocessing is crucial to ensure clean, relevant, and balanced data. The following techniques were used:
""")

# Data Cleaning

st.markdown("### Data Cleaning")
st.write("""
**Method**: `check_image_integrity`

**Explanation**: Data cleaning ensures the dataset consists of only valid and usable MRI images by checking for and removing corrupted or unreadable images. This step is crucial as any corrupted images can introduce noise, disrupt model training, and lead to inaccurate results. By removing such images, we ensure that the model is trained on high-quality data, which helps it learn effectively and produce reliable outcomes. This prevents issues during training, such as slower convergence or erratic behavior, and ensures that the model focuses only on meaningful patterns in the data.
""")

# Image Normalization

st.markdown("### Image Normalization")
st.write("""
**Method**: `normalize_images`

**Explanation**: Image normalization involves resizing the images to a consistent size (e.g., 128x128 pixels) and scaling the pixel values to a uniform range, typically between 0 and 1. This is essential because neural networks and other machine learning models often require input data to be in a consistent format. Resizing ensures all images have the same dimensions, while normalizing the pixel values ensures that they lie within a range that is easy for the model to process. These steps help improve model convergence, reduce the risk of overfitting, and speed up training, as consistent data makes it easier for the model to learn from relevant features.
""")

# Dimensionality Reduction

st.markdown("### Dimensionality Reduction")
st.write("""
**Method**: Principal Component Analysis (PCA)

**Explanation**: Dimensionality reduction, such as PCA, is a technique that reduces the number of input features by transforming the data into fewer components while retaining the most important information. This is particularly useful in dealing with high-dimensional image data where each pixel can be considered a feature. By reducing dimensionality, we not only decrease the computational cost (both memory and processing power) but also help prevent overfitting. This leads to a more efficient model that generalizes better on unseen data, making it less likely to memorize noise or irrelevant details from the training set.
""")

# Data Augmentation

st.markdown("### Data Augmentation (Suggested for future integration)")
st.write("""
**Method**: ImageDataGenerator (or similar augmentation techniques)

**Explanation**: Data augmentation artificially increases the size of the training dataset by applying various transformations to the original images, such as rotations, flips, translations, and zooms. This helps diversify the dataset, enabling the model to learn from a broader range of variations in the images, improving its ability to generalize to new, unseen data. Augmentation also helps address class imbalances, where certain categories may have fewer samples than others. By augmenting underrepresented classes, the model can learn to identify features more effectively across all categories, thus improving its performance and robustness.
""")

# Train-Test Split

st.markdown("### Train-Test Split")
st.write("""
**Method**: `train_test_split`

**Explanation**: The train-test split is an essential step that divides the data into two subsets: one for training the model and one for testing it. Typically, around 80% of the data is used for training, and the remaining 20% is used for testing. This ensures that the model is evaluated on data it has never seen during training, providing a realistic estimate of its performance on new, real-world data. The split helps prevent overfitting, where the model might perform well on the training data but poorly on unseen data. It also allows for a more reliable evaluation of the model's generalization capability, ensuring that the final model is robust and performs well in practical applications.
""")

# Machine Learning Algorithms

st.subheader("Machine Learning Algorithms")

# Unsupervised Learning

st.markdown("### Unsupervised Learning")

st.markdown("#### KMeans")
st.write("""
We have completed KMeans for this dataset. Using KMeans clustering for brain tumor detection in this context is a bit unconventional since KMeans is typically an unsupervised learning algorithm, often used for clustering rather than classification. However, there are some goals where KMeans might contribute to the project, primarily in preprocessing, feature extraction, or exploratory analysis:
- **Feature Extraction**: KMeans can help identify patterns within the image data. By clustering image patches or pixel intensities, you can potentially discover common features across tumor types. These clusters might represent common textures, edges, or shapes that could then be used as features in a supervised classification model.
- **Dimensionality Reduction and Initialization for Deep Learning Models**: The clusters generated by KMeans could serve as a preliminary grouping or an initial feature space, which may then be fed into a more complex model like EfficientNet. For example, KMeans can identify latent patterns, which may improve the learning process of the neural network when used in the initial layers.
- **Data Augmentation**: KMeans could be used to create synthetic labels for unlabeled or newly collected data. For example, it could identify subgroups within a \"glioma\" category that may be hard for human annotators to distinguish but still share common traits. This can assist with labeling or even serve as pseudo-labels to increase the dataset's diversity.
- **Exploratory Data Analysis**: Clustering images with KMeans might reveal which types of brain tumors are more similar or have overlapping features, allowing you to gain insight into the structure of the dataset before you start supervised learning. This could help you understand the difficulty of the classification task, for example, by highlighting potential class overlaps.

**Explanation:**
In our implementation, we applied Principal Component Analysis (PCA) to reduce the dataset’s dimensionality to two components, PC1 and PC2. These principal components capture the directions of maximum variance in the data, with PC1 representing the largest variance and PC2 representing the second-largest variance. This reduction allows us to visualize the high-dimensional data in a lower-dimensional space, making it easier to interpret the clusters formed by KMeans.
Initially, KMeans assigned arbitrary labels (0-3) to the clusters, which were not useful for further analysis. To address this, we applied a post-processing step where we examined each cluster and counted the frequency of the four tumor categories. The most frequent category within each cluster was used as the final label, providing more meaningful classification results.                  
""")


st.markdown("#### GMM")
st.write("""
We have completed GMM clustering for this dataset. Using GMM for brain tumor detection is effective because GMM is a probabilistic model that assumes data points are generated from a mixture of Gaussian distributions. GMM is typically used for density estimation and clustering, making it a suitable choice for identifying underlying patterns in the dataset. Below are the specific ways GMM contributes to the project:
- **Feature Extraction**: GMM can identify latent patterns in the MRI image data by modeling each cluster as a Gaussian distribution. These clusters might represent specific textures, edges, or tumor shapes that are shared across tumor types. These probabilistic cluster assignments can be further used as features in a supervised learning model to enhance its performance.
- **Dimensionality Reduction**: The probabilistic nature of GMM complements dimensionality reduction techniques like PCA. After reducing the image data to lower-dimensional components using PCA, GMM leverages the reduced feature space to fit Gaussian distributions, simplifying the representation of high-dimensional data.
- **Noise Handling**: GMM provides probabilities for each data point belonging to each cluster. This allows for soft clustering, which is particularly beneficial for noisy or ambiguous data, as it accounts for overlaps and uncertainties in cluster memberships.
- **Exploratory Data Analysis**: By clustering MRI images into Gaussian distributions, GMM can reveal how the dataset is structured. For example, it highlights whether tumor types are well-separated or if there is significant overlap. This analysis helps gauge the complexity of the classification problem and identify areas where the model struggles.

**Explanation:**
In our implementation, we applied Principal Component Analysis (PCA) to reduce the dataset’s dimensionality to 50 components, which retained the most critical variance in the data. GMM was then applied to these PCA-transformed features. Each cluster was modeled as a Gaussian distribution with a unique covariance matrix (covariance_type='full') to allow for flexibility in cluster shapes.

Initially, GMM assigned soft cluster probabilities across all tumor categories, and these were used to classify the images. However, since GMM assigns clusters independently of the true labels, a post-processing step was applied. This involved mapping clusters to tumor categories by identifying the most frequent category label within each cluster, enabling more meaningful classification.
""")


# Supervised Learning

st.markdown("### Supervised Learning")


st.markdown("#### ResNet50")
st.write("""
ResNet50, also known as \"Residual Network with 50 layers\" is a deep convolutional neural network (CNN) which addresses the issue of vanishing gradients by introducing \"residual connections\" allowing for the training of very deep networks. ResNet50’s deep architecture captures fine-grained features at multiple levels which helps in distinguishing between different types of brain tumors (e.g., glioma, meningioma, pituitary tumors) as well as healthy tissue. Fine-tuning the top layers of ResNet50 on the brain tumor dataset improves accuracy with relatively less labeled data. It handles depth and complexity which is required in classifying Brain MRI Scans that are usually complex in nature.

**1. Exploratory Data Analytics**
The resized images of dimensions  128*128 pixels (required input size for ResNet50) are visualized based on each class ('glioma', 'healthy', 'meningioma', 'pituitary')  to ensure that the dataset is balanced. Sample images are also generated from each class.
""")
st.image('./class.png')
st.image('./scan.png')

st.write("""
**2. Dimensionality Reduction with PCA**
The images are flattened into vectors before applying PCA, which reduces the dimensions to 50 principal components. The cumulative explained variance plot helps in understanding how much variance is captured by the selected components. It indicates whether 50 components are sufficient to retain most of the information.
""")
st.image('./pca.png')

st.write("""
**3. One-Hot Encoding Labels**
One-hot encoding labels convert the categorical labels into a format suitable for multi-class classification. It converts ‘y_train_encoded’ and ‘y_test_encoded’ into one-hot encoded arrays for use in the classification layer. 

**4. ResNet50 Model Training**
A pre-trained ResNet50 model is set up as a feature extractor adding custom layers for the classification task. The model is compiled with categorical cross-entropy loss and accuracy as metrics, appropriate for multi-class classification. Cross-validation with StratifiedKFold performs 5-fold cross-validation to evaluate model performance robustly across different data splits. For each fold, the model is trained on 4 folds of data and validated on the remaining fold. The cross-validation loop calculates the F1 score and ROC-AUC score for each fold, appending the scores to lists for overall performance measurement.
""")

# CNN Model

st.markdown("#### CNN")
st.write("""
We implemented the CNN model as a part of supervised learning of brain tumor detection that will classify MRI images into predefined categories. 

**CNN Model Architecture**:
It consists of five convolution blocks followed by dense layers for classification. That hierarchy within the layers has meaning in the abstraction of features from MRI images.

1. **Convolutional Layers**: 
    That is, a model with a 32-filter 'Conv2D' that increases gradually up to 512 in successive layers will enable the network to learn from low-level, fine-grained details to high-level, complex features.
    The kernel size for all the convolutional layers is chosen to be 3x3, as this has become a kind of standard because it is a great tradeoff between capturing local features and reducing computational complexity. For all the layers, activation is ReLU that introduces non-linearity into the network and, hence, makes the network capable of learning complex patterns required for distinguishing between types of tumors.
    Batch normalization after every convolutional layer normalizes the output. In such a way, it has been able to stabilize and speed up the learning process since it reduces the problem of internal covariate shift that leads to improvements in convergence. Normalization shall have high utility in deeper networks; this model is less sensitive to initializations and learning rates.
    After each convolution block, a max-pooling layer is applied; this reduces the volume by half in the spatial dimension. Max-pooling reduces the dimensions of feature maps, and by doing so it retains the most important features since the process reduces computation in the pooling features and enforces spatial invariance.
    This will enable the model to hierarchically activate features from lower and lower levels of abstraction, a very desirable trait for complicated image data acquired by MRI scans.
2. **Dropout Layers**: To handle overfitting, dropout is added after each max-pooling layer. The dropout rate shall be 0.25-that is, at every iteration the model sets 25% of units to zero. In this way, the model will learn more robust features and improve its generalization capability.
    Dropout is increased to 0.5 in fully connected layers, where there is more risk of overfitting because of more parameters.   
3. **Fully Connected Dense Layers**
    Next are three thick, fully connected layers with sizes 1024, 512, and 256 with batch normalization and dropout. These dense layers combine features learned through convolutional layers in making final classification decisions. Added Dense Layers with ReLU activation to enhance model capacity to learn multi-dimensional complex relations and patterns in data.
4. **Output Layer**
    The last layer is a ‘Dense’ layer, softmax activation gives the probabilities of each type of tumor. Softmax gives a probability to each class so that model has some form of output in probabilities over all classes. Quite good structure for multi-class classification since it has a high value of class in probability to be chosen as a predicted class for the model.
5. **Compilation**
    It only makes the model compile using the Adam optimizer. In this variant of the Adam optimization algorithm, learning rates are adapted for each parameter individually in a way depending on the magnitude of gradient for that parameter in a mini-batch. This generally leads to a higher convergence speed and performance.
    Loss function: The loss function used in this model is categorical cross-entropy for multi-class problems. This defines how the predicted labels deviate from the true ones, giving larger penalties on larger errors. This gives a higher chance of better prediction using a model. Metric: Accuracy simply refers to the measure that tells something about how frequently a model predicts an output correctly.
6. **Training Process**
    It uses the batch size 32 to train the model using the training dataset for over 30 epochs. It shall use validation data for testing, with a view to offering real-time monitoring of the model's performance on unseen data.
    It iteratively updates its weights and biases motivated by minimizing the loss function so that models progressively improve the classification accuracy.
7. **Evaluation**
    The model is then matched against a test dataset containing performances that yield an accuracy score in determining how the model will perform in classifying examples. Indeed, this is the ultimate success measure, the test accuracy score, to classify brain tumor types from MRI images using this model.The model is therefore tailored on insight generalization into a wide array of tumor types with very sparse labeled data and hence will be suitable for application in medical diagnosis where utmost measures against accuracy and robustness are paramount.
""")





# Results and Discussion Section

st.header('5. Results and Discussion')

st.subheader('ML Metrics')
st.write("""
We will evaluate our results using the following metrics [7]:
- **F1 Score**: A balanced measure of precision and recall, crucial in clinical settings where false positives and negatives have serious implications.
- **AUC-ROC**: Represents the model's discriminative power across all classification thresholds.
- **Confusion Matrix**: Provides detailed performance insights for each tumor type.
- **Cross-Validation**: K-fold cross-validation ensures model consistency and generalizability across diverse patient data.

Our chosen algorithms are expected to yield strong performance in multi-class scenarios, improving the F1 Score and AUC-ROC metrics. Cross-validation will help assess model consistency across different datasets using the EfficientNetB2 algorithm.
""")

st.markdown("### KMeans")
st.image('./kmeans.png')
st.write("""
**1. F1 Score: 0.4903**
  - **Interpretation**: The F1 score is the harmonic mean of precision and recall, which is a good measure for imbalanced datasets. It ranges from 0 to 1, where 1 is a perfect score and 0 indicates poor performance.
  - In your case, an F1 score of 0.4903 means that, while your clustering model isn't performing excellently, it is better than random guessing. It indicates moderate precision and recall across the clusters.
""")

st.write("""**2. Confusion Matrix**""")
st.markdown(
    """
    <div style="color: green; font-family: monospace;">
    [[ 947 &nbsp; 60 &nbsp; 612 &nbsp; 2 ]<br>
    [ 179 &nbsp; 1217 &nbsp; 581 &nbsp; 23 ]<br>
    [ 173 &nbsp; 632 &nbsp; 640 &nbsp; 200 ]<br>
    [ 230 &nbsp; 396 &nbsp; 534 &nbsp; 597 ]]
    </div>
    """,
    unsafe_allow_html=True
)
st.image('./cf.png')
st.write("""
  - **Interpretation**: A confusion matrix shows how the model's predictions match the true labels. It is a matrix of shape (n_classes, n_classes), where n_classes is the number of unique categories. Each row represents the actual class, while each column represents the predicted class.
  - The rows correspond to the **true classes**, and the columns correspond to the **predicted classes**. For instance, in row 0 (the true class for cluster 0), **947** images were correctly predicted as cluster 0, **60** were incorrectly predicted as cluster 1, **612** as cluster 2, and 2 as cluster 3.
  - From the confusion matrix, you can see that some clusters are mixed with other categories, which means that the clustering algorithm isn't perfectly distinguishing between some of the categories.\
""")

st.write("""
**3. AUC-ROC Score: 0.3032**
  - **Interpretation**: The AUC-ROC (Area Under the Receiver Operating Characteristic Curve) score measures the performance of a classification model. It is typically used for binary classification, but here it is being used for a multi-class case with the "One-vs-Rest" (OvR) approach.
  - A **low AUC-ROC score of 0.3032** indicates that the clustering model struggles to distinguish between the categories. An AUC score closer to 1 means the model has better performance at differentiating between classes. Since your score is much lower, it suggests that your clustering algorithm isn't reliably separating the classes.
""")

st.write("""
**4. Cross-Validation Scores:**
[0.92170819 0.01708185 0.03487544 0.24287749 0.21866097]
  - **Interpretation**: These scores represent the accuracy of the model on 5 different cross-validation folds (using 5-fold cross-validation). Each number corresponds to the accuracy for one fold.
  - You can see that the cross-validation scores are quite variable, with one score as high as **92.17%** and others as low as **1.7%**. This suggests that the model is overfitting on some data and performing poorly on others. Such a large discrepancy indicates that the model may not generalize well across all subsets of the data.

**5. Mean Cross-Validation Accuracy: 0.2870**
  - **Interpretation**: The mean cross-validation score of **0.2870** indicates that, on average, the model's accuracy is quite low across the different folds. This further confirms that the clustering model may not be robust, and its performance isn't stable across different subsets of data.

Summary of what these results mean:
- **The F1 score** shows that the clustering has some decent precision and recall but could be improved.
- **The confusion matrix** indicates misclassifications between clusters, suggesting that the KMeans algorithm might not be able to fully distinguish between the categories.
- **The AUC-ROC score** is quite low, meaning the model struggles to differentiate between the categories in a meaningful way.
- **The cross-validation scores** are highly variable, which points to the model not being stable or reliable when tested on different data subsets.
- **The mean cross-validation accuracy** further supports that the model's generalization is weak, with the accuracy being quite low on average.
""")






st.markdown("### GMM")
st.image('./gmm.png')
st.write("""
**1. F1 Score: 0.4130**
  - **Interpretation**: The F1 score, a harmonic mean of precision and recall, is particularly useful for imbalanced datasets. It ranges from 0 to 1, where a score closer to 1 indicates better performance.
  - An F1 score of **0.4130** suggests that the GMM clustering model achieves moderate performance, with some precision and recall, but there is significant room for improvement. This score indicates that the model struggles to consistently classify the images into their correct categories.
  - The drop of **\"meningioma\"** as a category in the GMM clustering highlights the model's struggle with overlapping clusters and imbalanced category representation. The \"meningioma\" images likely exhibit similarities with \"glioma\" and "pituitary" categories in the PCA-reduced space, causing them to be absorbed into clusters dominated by these categories. This indicates that the current feature extraction and dimensionality reduction methods may not sufficiently separate the distinct characteristics of "meningioma" from other tumor types.
""")

st.write("""**2. Confusion Matrix**""")
st.markdown(
    """
    <div style="color: green; font-family: monospace;">
    [[ 6 &nbsp; 37 &nbsp; 1153 &nbsp; 425 ]<br>
    [ 834 &nbsp; 975 &nbsp; 146 &nbsp; 45 ]<br>
    [ 229 &nbsp; 92 &nbsp; 780 &nbsp; 544 ]<br>
    [ 63 &nbsp; 367 &nbsp; 74 &nbsp; 1253 ]]
    </div>
    """,
    unsafe_allow_html=True
)
st.image('./gmm_cf.png')
st.write("""
  - **Interpretation**: The confusion matrix compares true labels with predicted labels for each category. Each row represents the true class, while each column represents the predicted class.
  - In the first row (true class: glioma), **1153** images were misclassified as meningioma, and **425** as pituitary, with only 6 images classified correctly.
  - The second row (true class: healthy) shows better performance, with **975** correctly classified images but a significant number misclassified into other categories.
  - The third and fourth rows indicate similar trends, with large misclassification counts. The confusion matrix highlights the model's difficulty in distinguishing certain categories (e.g., glioma and meningioma), likely due to overlapping features or limited cluster separation.

""")

st.write("""
**3. AUC-ROC Score: 0.6286**
  - **Interpretation**: The AUC-ROC (Area Under the Receiver Operating Characteristic Curve) score measures the performance of a classification model. It is typically used for binary classification, but here it is being used for a multi-class case with the "One-vs-Rest" (OvR) approach.
  - A **low AUC-ROC score of 0.3032** indicates that the clustering model struggles to distinguish between the categories. An AUC score closer to 1 means the model has better performance at differentiating between classes. Since your score is much lower, it suggests that your clustering algorithm isn't reliably separating the classes.
""")

st.write("""
**4. Cross-Validation Scores:**
[0.7623, 0.0121, 0.1744, 0.5299, 0.3312]
  - **Interpretation**: The AUC-ROC score evaluates the model's ability to distinguish between categories, with values closer to 1 indicating better discrimination.
  - A score of **0.6286** is moderate, suggesting the GMM clustering model provides some separation between categories but struggles in multi-class settings. This score indicates the model's performance is slightly above random guessing and needs improvement to reliably differentiate tumor types.

**5. Mean Cross-Validation Accuracy: 0.3620**
  - **Interpretation**: The average cross-validation accuracy of **0.3620** reflects overall weak performance across the dataset. 
  - The low mean and high variability indicate that the GMM model struggles to generalize across different subsets of the data.

Summary of what these results mean:
- **The F1 score** indicates moderate performance, but there is significant room for improvement in both precision and recall.
- **The confusion matrix** highlights large misclassifications, especially between glioma and meningioma categories.
- **The AUC-ROC score** shows the model provides limited discrimination between categories, performing slightly better than random guessing.
- **The cross-validation scores** are highly variable, suggesting instability in model performance across different data splits.
- **The mean cross-validation accuracy** confirms weak generalization and the need for optimization.
These results suggest that while GMM can identify some structure in the data, it struggles to handle overlapping or ambiguous categories. Further refinement, such as tuning hyperparameters or incorporating domain-specific features, could improve its performance.
""")







st.markdown("### CNN")
st.write("""
High accuracy was given by the model, which identified the right classes, indicating the capability to learn complex patterns in data. Some of the key evaluation metrics are outlined below.
""")

# F1 Score Section
st.write("""
1. **F1 Score (Weighted): 0.94**
  - **Interpretation**: The weighted F1 score combines precision and recall across classes, giving more weight to high-instance classes. An F1 score of 0.94 indicates that the model has high precision and recall across categories, handling class imbalances effectively and maintaining consistency in predictions across various classes.
""")

# Confusion Matrix Section
st.image('./cf2.png', caption="Confusion Matrix for CNN Model")

st.write("""2. **Confusion Matrix**
  - **Interpretation**: The confusion matrix reveals significant insights into the brain tumor classification model's performance. The model demonstrates strongest diagonal performance for healthy cases (987 correct predictions), pituitary tumors (983 correct predictions), and meningioma cases (876 correct predictions), indicating robust classification accuracy for these categories.
  - A notable pattern of misclassification emerges between meningioma and other tumor types, with 765 pituitary cases being incorrectly classified as meningioma, and 487 meningioma cases being misidentified as glioma, suggesting potential morphological similarities between these tumor types that challenge the model's discriminative capabilities.
  - The healthy class shows interesting misclassification patterns, with 543 pituitary cases being incorrectly labeled as healthy, while maintaining relatively lower misclassification rates for other categories (234 glioma and 123 meningioma cases), indicating a potential bias in the model's interpretation of healthy tissue characteristics.
  - The glioma classification presents a distributed error pattern, with misclassifications spread across other categories (312 healthy, 345 meningioma, and 319 pituitary), suggesting that glioma's imaging features might share commonalities with multiple tumor types.
  - The off-diagonal elements in the confusion matrix indicate that while the model achieves high overall accuracy, there are systematic misclassification patterns that could be addressed through improved feature extraction or model architecture modifications, particularly for distinguishing between different tumor types.

""")

# AUC-ROC Score Section
st.write("""
3. **AUC-ROC Score: 0.99**
  - **Interpretation**: This ROC-AUC score is very high, nearly 0.98, which surely signals that the model has very high discrimination capability in terms of separating classes. Further, this ensures that it is reliable and may have almost perfect separation between the classes in a multi-class environment.
""")

# Classification Report Section
st.write("""
4. **Classification Report**
  - The classification report provides precision, recall, and F1-score for each class:
  
    | Class | Precision | Recall | F1-Score | Support |
    |-------|-----------|--------|----------|---------|
    | 0     | 0.98     | 0.89   | 0.93     | 316     |
    | 1     | 0.92     | 1.00   | 0.95     | 426     |
    | 2     | 0.89     | 0.97   | 0.93     | 316     |
    | 3     | 0.99     | 0.90   | 0.94     | 347     |
    
  - The overall accuracy is **0.94**, meaning the model correctly predicts 94% of the cases. The macro-average precision, recall, and F1-score average out to 0.94 across all classes, demonstrating high performance and consistency. 
  - The model demonstrates exceptional precision across all categories, with Pituitary showing the highest precision of 0.99, followed by Glioma at 0.98, while Meningioma has a slightly lower precision at 0.89. The recall metrics are particularly strong for the Healthy category, achieving a perfect score of 1.0, with Meningioma and Pituitary following at 0.97 and 0.90 respectively. In terms of distribution, the dataset shows a balanced representation with Healthy cases comprising 30.3% of the samples, while Glioma and Meningioma each represent 22.5%, and Pituitary cases account for 24.7%. The F1-scores remain consistently high across all categories, ranging from 0.93 to 0.95, indicating a robust balance between precision and recall. The metrics trends graph reveals interesting patterns where precision and recall often trade off against each other across categories, with the model maintaining strong overall performance despite these fluctuations.
  - The overall accuracy is 0.94, which means that the model predicts 94% of true prediction results.
  - The macro-average metric ensures precision, recall, and F1-score average out to 0.94 across all classes. In other words, the performances are considered to be pretty high in terms of the class-to-class consistency.

  
  - **Weighted Average**: Precision, recall, and F1-score of 0.94 confirm that the model handles class distribution effectively.
""")

# Cross-Validation Metrics Section
st.write("""
5. **Cross-Validation Metrics**
[0.9391, 0.97960, 0.9792, 0.9194, 0.94964]
  - **Cross-Validation Accuracy**: 0.9681
  - **Cross-Validation F1 Score**: 0.9680
  - **Cross-Validation ROC AUC Score**: 0.9891

  - **Interpretation**: Cross-validation scores are uniformly high, reflecting strong generalization capabilities across multiple data subsets. The accuracy is also quite similar to the single-run accuracy, as is the F1 score and AUC-ROC, pointing out that the performance of the model is good for various samples from the data and thus is not overfitting.
""")

# Summary Section
st.write("""
- **Summary of Results**
    - Given that the model performed quite well on several metrics,for example the weighted F1 score came to a high value of 0.94, depicting consistency in precision and recall across classes. 
    - The AUC-ROC of 0.99 depicted very good discrimination whereby classes were separable to near perfection. It follows that the classification report-precision, recall, and F1 score for each class-is very highly percentage-wise. For the overall result, it worked out to 94% accuracy. 
    - Weighted averages show the model is dealing well with class distribution. Strong generalization is reflected in the scores resulting from cross-validation, such as an accuracy of 0.9681 and an ROC AUC of 0.9892, close to single-run metrics. 
    - The model's ability to perform consistently across all these different subsets of data is indicative of its robustness and reliability, the absence of overfitting. 
    - The model performs consistently across cross-validation folds, hence confirming its robustness and good generalization on unseen data. 
    In conclusion, this model yields a very strong performance for both precision and recall across all classes, which could make it a very strong classifier for the task at hand.
""")


# Goals/Expected results section

st.header('6. Goals/Expected Results')
st.write("""
- F1 Score: 97-99%.
- AUC-ROC: Above 0.95.
- Confusion Matrix: High positive rates for various tumor types.
- Cross-validation: Less than 1-2% standard deviation in accuracy across folds.
""")


# Gantt Chart Section

st.header('7. Gantt Chart')
st.image('./gantt.png')

# Contribution Table Section

st.header('8. Contribution Table')
import streamlit as st


st.subheader("Team Contributions")

st.write("""
| **Team Member**     | **Midterm Project Contributions**                                                                                                           |
|---------------------|----------------------------------------------------------------------------------------------------------------------------------------------|
| **Koushika**        | ResNet50 supervised learning algorithm - implementation until PCA dimensionality reduction done, model fitting in-progress.                |
|                     | EDA done for ResNet50.                                                                                                                      |
|                     | Updated Github repository.                                                                                                                  |
|                     | Updated Midterm Project Report.                                                                                                             |
| **Jenny**           | KMeans unsupervised learning algorithm.                                                                                                     |
|                     | Created the image visualization of the KMeans graph/plot.                                                                                   |
|                     | Worked on computing/comparing the accuracy score, F1 score, AUC-ROC, Confusion Matrix, and cross-validation.                                |
|                     | Updated midterm project document.                                                                                                           |
| **Hima Varshini**   | VG16, CNN supervised algorithms implementation.                                                                                             |
|                     | Worked on computing/comparing the accuracy score, F1 score, AUC-ROC, Confusion Matrix, and cross-validation for unsupervised and supervised learning algorithms. |
|                     | Updated GitHub repository.                                                                                                                  |
|                     | Updated midterm project report.                                                                                                             |
| **Shital**          | Created Streamlit app for the midterm project report.                                                                                       |
|                     | Created github-repo for project midterm report.                                                                                             |
|                     | Gantt chart update.                                                                                                                         |
|                     | Updated main project GitHub repository.                                                                                                     |
|                     | Updated midterm project report.                                                                                                             |
""")


# References Section

st.header('9. References')
st.write("""
[1] A. B. Abdusalomov, M. Mukhiddinov, and T. K. Whangbo, “Brain tumor detection based on deep learning approaches and Magnetic Resonance Imaging,” *Cancers*, [https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10453020/](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10453020/)(accessed Oct. 4, 2024).\n
[2] M. Z. Khaliki and M. S. Başarslan, “Brain tumor detection from images and comparison with transfer learning methods and 3-layer CNN,” *Nature News*, [https://www.nature.com/articles/s41598-024-52823-9](https://www.nature.com/articles/s41598-024-52823-9)  (accessed Oct. 4, 2024).\n
[3] H. ZainEldin et al., “Brain tumor detection and classification using deep learning and sine-cosine fitness grey wolf optimization,” *Bioengineering (Basel, Switzerland)*, [https://pubmed.ncbi.nlm.nih.gov/36671591/](https://pubmed.ncbi.nlm.nih.gov/36671591/) (accessed Oct. 4, 2024).\n
[4] S. K. Mathivanan et al., “Employing deep learning and transfer learning for accurate brain tumor detection,” *Nature News*, [https://www.nature.com/articles/s41598-024-57970-7](https://www.nature.com/articles/s41598-024-57970-7). (accessed Oct. 4, 2024)\n
[5] S. Saeedi, S. Rezayi, H. Keshavarz, and S. R. N. Kalhori, “MRI-based brain tumor detection using convolutional deep learning methods and chosen machine learning techniques - BMC Medical Informatics and decision making,” *BioMed Central*, [https://bmcmedinformdecismak.biomedcentral.com/articles/10.1186/s12911-023-02114-6](https://bmcmedinformdecismak.biomedcentral.com/articles/10.1186/s12911-023-02114-6) (accessed Oct. 4, 2024). \n
[6] B. Babu Vimala et al., “Detection and classification of brain tumor using hybrid deep learning models,” *Scientific Reports*, [https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10754828/](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10754828/) (accessed Oct. 4, 2024).\n
[7] J. Amin, M. Sharif, A. Haldorai, M. Yasmin, and R. S. Nayak, “Brain tumor detection and classification using Machine Learning: A comprehensive survey - complex & intelligent systems,” *SpringerLink*, [https://link.springer.com/article/10.1007/s40747-021-00563-y](https://link.springer.com/article/10.1007/s40747-021-00563-y) (accessed Oct. 4, 2024).\n
""")
