Guitar Chord Predictor:

This project contains 3 machine-learning models that classify an input audio signal of a guitar chord into 10 different guitar chords: A, Am, B, C, D, Dm, E, Em, F, and G. The three 
models used in this project are the Decision Tree Classifier, Support Vector Machine(SVM), and the Convoluted Neural Network(CNN) model, along with some ensemble methods, and hyperparameter
tuning to boost the performance of the models. The audio input was preprocessed using Librosa to extract the chroma vectors, which were padded to convert into a numpy array to allow 
training on different models. The extracted chroma vectors encode information corresponding to the 12 chromatic scales, which allows the CNN to work efficiently due to the structure
of the data.

After running the three models, the most effective model for guitar chord identification was the CNN model with an accuracy of 95.3% on the testing set. The SVM classifier, after the hyperparameter
tuning using GridSearchCV() was also equally effective with an accuracy of 94.3% on the testing set, while the decision tree model with ensemble method, random forest classifier, performed
slightly weaker with a testing set accuracy of 88.8%.

Required modules:

    numpy
    librosa
    tensorflow
    scikit-learn
    matplotlib

To run the jupyter notebook containing the model, run the following commands to load data.json before running the jupyter notebook:

    python data_loader.py
