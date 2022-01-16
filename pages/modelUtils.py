import pickle
import os
import sklearn
from sklearn import linear_model

def checkDir():
	if 'models' in os.listdir('../'): 
		return True
	return False

def makeDir():
	if checkDir(): 
		pass
	else: 
		os.mkdir('../models')

# guardará un modelo en ../models y devolverá la ubicación + nombre del modelo guardado
def saveModel(modelClass, name = None):
	fileName = name
	if name is None: 
		fileName = 'model'+str(len(os.listdir('../models')))
	fileName+='.sav'
	pickle.dump(modelClass, open('../models/'+fileName, 'wb'))
	return '../models/'+fileName

# modelo se cargará a través de la ubicación del modelo que se devuelve desde el 
def loadModel(fileName):
	model = pickle.load(open(fileName, 'rb'))
	return model

# Todas las pruebas siguientes pasaron
if __name__ == '__main__':
	print(checkDir())
	makeDir()
	reg = linear_model.Ridge(alpha = 0.5)
	reg.fit([[0, 0], [0, 0], [1, 1]], [0, .1, 1])
	print("og Coeff: ",reg.coef_)
	path = saveModel(reg)
	print("Model Name: "+path)
	model = loadModel(path)
	print("Loaded Model:", model.coef_)
