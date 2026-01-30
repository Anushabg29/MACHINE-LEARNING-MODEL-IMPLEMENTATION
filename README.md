TASK 4 # MACHINE-LEARNING-MODEL-IMPLEMENTATION

*COMPANY* : CODTECH IT SOLUTIONS

*NAME* : ANUSHA B G

*INTERN ID* : CTIS1977

*DOMAIN* : PYTHON PROGRAMMING

*DURATION* : 4 WEEKS

*MENTOR* : NEELA SANTOSH

PROJECT OVERVIEW

This project focuses on building a Spam Message Detection System using Machine Learning and Natural Language Processing (NLP) techniques. The main objective of the system is to automatically classify text messages as either Spam or Ham (Not Spam). Spam messages often contain promotional, fraudulent, or misleading content, while ham messages represent normal and genuine communication.

The system uses a supervised learning approach, where a labeled dataset containing spam and ham messages is used to train a classification model. The model learns patterns in the text data and predicts whether new, unseen messages are spam or not. This project is implemented using Python and popular machine learning libraries such as pandas and scikit-learn.

STEP BY STEP CODE OVERVIEW

Importing Required Libraries
The code begins by importing essential Python libraries.

pandas is used for data loading and manipulation.

scikit-learn modules are used for data splitting, text vectorization, model training, and evaluation.

Loading the Dataset
The dataset (spam.csv) is loaded using pandas.read_csv().
A debug print statement is included to confirm that the dataset has been loaded successfully and to check the number of rows.

Data Preprocessing

The column names are renamed to label and message for clarity.

Missing values are removed to avoid errors during training.

The labels are cleaned by converting them to lowercase and removing extra spaces.

Labels are encoded numerically:

ham → 0

spam → 1

Dataset Validation
A safety check ensures that the dataset is not empty. If the dataset is empty, the program raises an error, preventing incorrect model training.

Splitting the Dataset
The dataset is divided into training and testing sets using train_test_split().

80% of the data is used for training.

20% is used for testing.
This helps evaluate how well the model performs on unseen data.

Text Vectorization
Machine learning models cannot work directly with text, so the messages are converted into numerical features using CountVectorizer.

Stop words such as “is”, “the”, and “and” are removed.

Each message is transformed into a bag-of-words representation.

Model Training
A Multinomial Naive Bayes classifier is used to train the model.
This algorithm is especially suitable for text classification problems because it works well with word frequency data.

Prediction and Evaluation

Predictions are made on the test dataset.

Model performance is evaluated using:

Accuracy Score

Confusion Matrix

Classification Report (Precision, Recall, F1-score)

Custom Message Testing
The trained model is tested with custom messages to demonstrate real-world usage.
The system predicts whether each message is SPAM or HAM.

FEATURESS OF THE SYSTEM

Automatic classification of messages into Spam or Ham

Efficient preprocessing and cleaning of text data

Uses NLP techniques for feature extraction

Reliable and fast prediction using Naive Bayes algorithm

Supports testing with custom user messages

Provides detailed evaluation metrics for model performance

PLOT FORM USED

In this project, no graphical plots are used.
The results are evaluated using text-based metrics such as accuracy score, confusion matrix, and classification report. These metrics provide clear numerical insights into model performance without the need for visual plots.

LANGUAGES AND TECHNOLOGIES USED

Programming Language: Python

Libraries and Frameworks:

pandas

scikit-learn

CountVectorizer (NLP)

Multinomial Naive Bayes

Machine Learning Type: Supervised Learning

Domain: Natural Language Processing (NLP)

Platform Used

Python IDLE / Jupyter Notebook / VS Code

Can be executed on Windows, Linux, or macOS

Requires Python 3.x environment

APPLICATIONS OF THE PROJECT

Email spam filtering systems

SMS spam detection

Social media message moderation

Customer support message filtering

Cybersecurity and fraud prevention

Content moderation in messaging applications

COMCLUSION

This Spam Message Detection project demonstrates how machine learning and NLP can be effectively used to solve real-world problems. By using a simple yet powerful algorithm like Naive Bayes, the system achieves accurate classification of spam messages. The project is easy to understand, scalable, and suitable for beginners as well as academic projects. It can be further enhanced by using advanced NLP techniques or deep learning models in the future.
