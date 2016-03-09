from Identification.LoadDescriptors import load_all_descriptors
from Identification.PreprocessingDescriptors import preprocess
from Identification.TrainCvTest import separate_databases
from FScore import F1Score

Descriptors = load_all_descriptors()
normalized_features, yClass, features_names = preprocess(Descriptors)
del Descriptors  # Ya no lo voy a utilizar
normalizedTrain, yTrain, normalizedCV, yCV, normalizedTest, yTest = separate_databases(normalized_features, yClass)

F1 = F1Score(normalizedTrain, yTrain)  # entrena el algoritmo y lo mide sobre la misma base de datos

# execfile("testC-gamma.py")
# execfile("testDataSize.py")
# execfile("testFeatures.py")
# execfile("testPca.py")
# execfile("Classifier.py")


