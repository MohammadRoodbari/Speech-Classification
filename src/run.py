import pandas as pd
from sklearn.model_selection import train_test_split

from steps.FeatureExtraction import FeatureExtractionPipeLine
from steps.DataPreProcessing import DataPreProcessing

from models.svc import SVMClassifier , Evaluation
from models.cnn import CNNModel

feature_extractor = FeatureExtractionPipeLine('spectral_centroid',
                                              'spectral_bandwidth',
                                              'rolloff',
                                              'melspectrogram',
                                              'spectral_contrast',
                                              'spectral_flatness',
                                              'mfcc')
feature_extractor.transform('D:\Speech processing\Speech Classification and Clustering\data')

pre_processor =  DataPreProcessing(r'D:\Speech processing\Speech Classification and Clustering\data\dataV2.csv',
                                   r'D:\Speech processing\Speech Classification and Clustering\src\dataset\voice_data.csv')

Dataset = pre_processor.transform()

# Dataset = pd.read_csv('D:\Speech processing\Speech Classification and Clustering\src\dataset\Data_final.csv' , low_memory= False)

########## Gender Classification ##########

X=Dataset.drop(['sex' , 'voice_id' , 'emotionID','textID','age','Unnamed: 0'],axis=1)
y=Dataset['sex']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3 , random_state= 42 , shuffle=True)

svm_classifier = SVMClassifier(kernel='rbf',C=150)
svm_classifier.train(X_train,y_train)
y_pred_test = svm_classifier.predict(X_test)

evaluator = Evaluation(y_test,y_pred_test)

print("Train_accuracy    : {:.4f}%" . format(Evaluation(y_train, svm_classifier.predict(X_train)).accuracy()*100))
print("Test_accuracy     : {:.4f}%" . format(evaluator.accuracy()*100))
print("precision_score     : {:.4f}%" . format(evaluator.precision()*100))
print("f1_score          : {:.3f}%" . format(evaluator.f1()*100))
print(evaluator.ConfusionMatrix())

########## Emotion Classification ##########

X=Dataset.drop(['sex' , 'voice_id' , 'emotionID','textID','age','Unnamed: 0'],axis=1)
y=Dataset['emotionID']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3 , random_state= 42 , shuffle=True)

num_classes = 4  # Number of classes 
model = CNNModel(num_classes)

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, pd.Series(list(map(lambda label: label-1 , y_train))), epochs=20, batch_size=32)

# test the model
results = model.evaluate(X_test,  pd.Series(list(map(lambda label: label-1 , y_test))), verbose = 0)
print('test loss, test accuracy:', results)