########################
# Universidad de Antioquia
# Procesamiento Digital de Imágenes
# Autores: Néstor Rafael Calvo Ariza – nestor.calvo@udea.edu.co
#           Santiago Alexis Patiño Múnera – alexis.patino@udea.edu.co
# Semestre 2019-2
###########################


# Libraries
# Library that extract the patterns in the image using lbp
from skimage.feature import local_binary_pattern
# Library used to navigate with paths and access files
import os
# Library used to open images
from PIL import Image
# Library used for mathematical operation
import numpy as np
# Library used to store the features and labels in files that can be open later
import pickle
# Library used to train a Support Vector Machine classifier
from sklearn.svm import SVC
# Library used to train a Random Forest classifier
from sklearn.ensemble import RandomForestClassifier
# Library used to standardize the dataset
from sklearn.preprocessing import StandardScaler
# Library used to execution time
from time import time
# Library used to try every parameters and select the best combination
from sklearn.model_selection import GridSearchCV
# Library used to show the results as confusion matrix, roc_curve, Area Under the Curve
from sklearn.metrics import confusion_matrix, roc_curve, auc
# Library used to plot the results
import matplotlib.pyplot as plt


def read_images(path_images):
    # Function created to read all images in a path
    #   Parameters:
    #       path_images: Contains the path that is going to be analyzed

    # Empty array to store all the images
    array_images = []
    # For loop to check every file in the path
    for files in os.listdir(path_images):
        # Check if the file ends in .jpeg, .jpg, .png
        extension = Image.open(path_images + "\\" + files).format
        if extension == 'JPG' or extension == 'JPEG' or extension == 'PNG':
            # If the file is an image, store the path for that image
            array_images.append(path_images + "\\" + files)
    # Return the array that contains the path for all images
    return array_images


def extract(path_images):
    # Function created to extract the features of images in a path
    #   Parameters:
    #       path_images: Contains the path that is going to be analyzed

    # Call the function "read_images" to obtain all the images in that pat
    images = read_images(path_images)
    # Variable use to check if its the first time that we are going to add a feature
    flag_first = True
    # Empty array to store the features for all the images in the path
    array_return = []
    # For loop to check every image in the array
    for image in images:
        # Open the image
        im = Image.open(image)
        # Extract the patterns using lbp
        lbp = local_binary_pattern(im, 5, 8 * 5)
        # Choose the max value obtained as the bin's limit
        n_bins = int(lbp.max() + 1)
        # Create the histogram
        hist, _ = np.histogram(lbp, density=True, bins=n_bins, range=(0, n_bins))
        # If is the first time, create an empty array
        if flag_first:
            array_return = np.empty(hist.shape, int)
            # Change the flag to false
            flag_first = False
        # Stack histogram values for every image
        array_return = np.vstack((array_return, hist))
    # Remove the first row, because is the row that we create as empty
    array_return = np.delete(array_return, 0, 0)
    # Return the array with the features
    return array_return


def save_DB():
    # Function created to create and save the database after all the features are
    # extracted

    # Paths for the database
    melanoma_train_path = r"BDN\melanoma\train"
    queratosis_train_path = r"BDN\queratosis\train"
    melanoma_test_path = r"BDN\melanoma\test"
    queratosis_test_path = r"BDN\queratosis\test"

    # Extract the features from the train dataset and label the dataset (1:Melanoma, 0:Queratosis)
    melanoma_dataset_train = extract(melanoma_train_path)
    queratosis_dataset_train = extract(queratosis_train_path)
    melanoma_train_labels = np.ones(melanoma_dataset_train.shape[0])
    queratosis_train_labels = np.zeros(queratosis_dataset_train.shape[0])

    # Extract the features from the test dataset and label the dataset
    melanoma_dataset_test = extract(melanoma_test_path)
    queratosis_dataset_test = extract(queratosis_test_path)
    melanoma_test_labels = np.ones(melanoma_dataset_test.shape[0])
    queratosis_test_labels = np.zeros(queratosis_dataset_test.shape[0])

    # Create dictionaries to save the information in a pickle
    m_db_dic = {"features": melanoma_dataset_train, "labels": melanoma_train_labels}
    q_db_dic = {"features": queratosis_dataset_train, "labels": queratosis_train_labels}
    m_test_dic = {"features": melanoma_dataset_test, "labels": melanoma_test_labels}
    q_test_dic = {"features": queratosis_dataset_test, "labels": queratosis_test_labels}

    # Save training and test features in a Pickle file to use it later on
    with open('M_DB.pickle', 'wb') as f:
        pickle.dump(m_db_dic, f, pickle.HIGHEST_PROTOCOL)
    with open('Q_DB.pickle', 'wb') as f:
        pickle.dump(q_db_dic, f, pickle.HIGHEST_PROTOCOL)
    with open('M_DB_test.pickle', 'wb') as f:
        pickle.dump(m_test_dic, f, pickle.HIGHEST_PROTOCOL)
    with open('Q_DB_test.pickle', 'wb') as f:
        pickle.dump(q_test_dic, f, pickle.HIGHEST_PROTOCOL)


def train_function():
    # Function created to train the system, using the features that are
    # already extracted

    # Read training dataset
    with open('M_DB.pickle', 'rb') as f:
        m_db = pickle.load(f)
    with open('Q_DB.pickle', 'rb') as f:
        q_db = pickle.load(f)
    with open('M_DB_test.pickle', 'rb') as f:
        m_test_db = pickle.load(f)
    with open('Q_DB_test.pickle', 'rb') as f:
        q_test_db = pickle.load(f)

    # Separate the features and the labels for train in variables
    x_m = m_db["features"]
    y_m = m_db["labels"]
    x_q = q_db["features"]
    y_q = q_db["labels"]
    # Separate the features and the labels for test in variables
    x_m_test = m_test_db["features"]
    x_q_test = q_test_db["features"]
    y_m_test = m_test_db["labels"]
    y_q_test = q_test_db["labels"]

    # Put both class in the same array, the one that we will fit to the machine
    X = np.vstack((x_m, x_q))
    # Put both class in the same array, the one that the system will predict
    X_test = np.vstack((x_m_test, x_q_test))
    # Put both labels in the same array, the one that we will fit to the machine
    y = np.hstack((y_m, y_q))
    # Put both labels in the same array, the one that we will use to compare the predictions
    y_test = np.hstack((y_m_test, y_q_test))

    # Normalise test and train feature array, in order to obtain mean = 0 and unit variance
    X = StandardScaler().fit_transform(X)
    X_test = StandardScaler().fit_transform(X_test)

    # Support Vector Machine (SVM)
    print("------------------------------------------------")
    print("Fitting the classifier to the training set")
    print("------------------------------------------------")
    # Start the timer
    t0 = time()
    # Create the range of parameters that we will analyze
    param_grid_SVM = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
                      'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
    # Try all the different combinations of parameters and select the best one
    clf = GridSearchCV(
        SVC(kernel='rbf', class_weight='balanced'), param_grid_SVM
    )
    # Use the best parameters and train the model
    clf = clf.fit(X, y)
    # Print the time that took the system to be trained
    print("SVM trained in %0.3fs" % (time() - t0))
    # Check the score for the best parameters
    best_SVM = clf.best_score_
    # Show the score
    print("Score: ", best_SVM)

    # Random Forest
    # Start the timer
    t0 = time()
    # Create the range of parameters that we will analyze
    param_grid_RF = {'n_estimators': [100, 500, 1000, 2000, 5000]}
    # Try all the different combinations of parameters and select the best one
    clf_RF = GridSearchCV(
        RandomForestClassifier(random_state=0), param_grid_RF
    )
    # Use the best parameters and train the model
    clf_RF = clf_RF.fit(X, y)
    # Print the time that took the system to be trained
    print("Random Forest trained in %0.3fs" % (time() - t0))
    # Check the score for the best parameters
    best_RF = clf_RF.best_score_
    # Show the score
    print("Score: ", best_RF)
    print("----------------------------------------")
    # Compare classifier and select the best classifier with the best parameters
    if best_RF > best_SVM:
        print("Best classifier is Random Forest with an score of ", best_RF, " against ", best_SVM, " obtained in SVM")
        print(clf_RF.best_estimator_)
        # Call the function "plot_result" that show the roc_curve, confusion matrix, etc usign the best classifer
        plot_results(clf_RF, X_test, y_test)
    else:
        print("Best classifier is SVM with an score of ", best_SVM, " against ", best_RF, " obtained in Random Forest")
        print(clf.best_estimator_)
        # Call the function "plot_result" that show the roc_curve, confusion matrix, etc usign the best classifer
        plot_results(clf, X_test, y_test)


def plot_results(classifier, X_test, y_test):
    # Function created to show the results obtained
    #   Parameters:
    #       classifier: Contains the variable that stores the classifier after being fit
    #       X_test: Contains the dataset that will be used for testing
    #       y_test: Contains the true labels for the test dataset

    # Calculate the prediction from the test dataset
    y_predict = classifier.predict(X_test)
    y_score = classifier.decision_function(X_test)
    # Create the confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_test, y_predict).ravel()
    print("-------Confusion matrix-------------")
    print("True negative", tn)
    print("True positive", tp)
    print("False negative", fn)
    print("False positive", fp)
    # Test and predicted labels printed, to show which present mistakes
    print("Melanoma's true labels: ", y_test[0:15], " Queratosis true labels: ",  y_test[16:31])
    print("Melanoma's predicted labels: ", y_predict[0:15], " Queratosis predicted labels: ",  y_predict[16:31])

    # Extract the data need for the roc curve
    fpr, tpr, _ = roc_curve(y_test, y_score)
    # Generate the Area Under the Curve (auc)
    roc_auc = auc(fpr, tpr)
    # Initialize a plot figure
    plt.figure()
    # Set basic parameters for the plot like colors, line width and the label
    plt.plot(fpr, tpr, color='darkorange',
             lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    # Plot a line that will be the reference
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    # Set the limits of the chart
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    # Set labels for both axis
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    # Set title for the graph
    plt.title('ROC Curve')
    # Select the place where the legend will be
    plt.legend(loc="lower right")
    # Show the graph
    plt.show()


def main():
    # Function created to set the basic things of the code

    # Save the path of the project
    ROOT = os.path.dirname(os.path.abspath("extract_features.py"))
    # Variable used to check how many .pickle files where found
    found = 0
    # Check all the files in the main folder
    for files in os.listdir(ROOT):
        # Check if the file end in .pickle
        if os.path.splitext(files)[1] == ".pickle":
            # Check if the file has the name "M_DB" or "Q_DB" or "M_DB_test" or "Q_DB_test"
            if os.path.splitext(files)[0] == "M_DB" or os.path.splitext(files)[0] == "Q_DB" or os.path.splitext(files)[
                0] == "M_DB_test" or os.path.splitext(files)[0] == "Q_DB_test":
                # If there is a file that meets those requirements, we add 1 to the variable created before
                found += 1
    # If we don't found the four files that contain the dataset, we will generate the dataset
    if found != 4:
        print("---------Files missing, creating the dataset----------------")
        # Generate the dataset
        save_DB()
        print("---------Creation successful, proceeding with the training----------------")
        # Call the function to train the system
        train_function()
    # In case that we found the four files, we dont need to create the dataset again, just train the network.
    else:
        # save_DB()
        print("----------All the pickle files were found, proceeding with the training--------------")
        # Call the function to train the system
        train_function()


# Call the main function
main()
