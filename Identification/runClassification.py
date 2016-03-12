from Identification.LoadDescriptors import loadAllDescriptors
from Identification.PreprocessingDescriptors import preprocessDescriptors
from Identification.TrainCvTest import separateDatabases
from utils.FScore import F1Score

Descriptors = loadAllDescriptors()
normalized_features, yClass, features_names = preprocessDescriptors(Descriptors)
del Descriptors  # Ya no lo voy a utilizar
normalizedTrain, yTrain, normalizedCV, yCV, normalizedTest, yTest = separateDatabases(normalized_features, yClass)

F1 = F1Score(normalizedTrain, yTrain)  # entrena el algoritmo y lo mide sobre la misma base de datos

# execfile("testC-gamma.py")
# execfile("testDataSize.py")
# execfile("testFeatures.py")
# execfile("testPca.py")
# execfile("Classifier.py")


