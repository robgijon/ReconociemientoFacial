#
#Autor: Roberto Gijón Liétor
#
#Descripción: Este programa sirve para entrenar y guardar un modelo usando las imagenes de las caras capturadas de todas las personas conocidas
#


import cv2
import os
import numpy as np

directorioDatos = '.../Datos'
listadoPersonas = os.listdir(directorioDatos)													#Obtenemos la lista de las personas reconocidas
print('Lista de personas. ', listadoPersonas)													#Las mostramos por consola

labels = []
facesData = []
label = 0

for nameDir in listadoPersonas:																	#Leemos cada las personas
	directorioPersona = directorioDatos + '/' + nameDir
	print('Leyendo imágenes...')

	for fileName in os.listdir(directorioPersona):												#Leemos cada imagen de cada persona
		print(nameDir + '/' + fileName)
		labels.append(label)																	#Agregamos cada persona identificandola por su etiqueta (la primera persona tendrá 0, la segunda 1, ...)
		facesData.append(cv2.imread(directorioPersona + '/' + fileName, 0))						#Agregamos la imagen de dicha persona
		imagen = cv2.imread(directorioPersona + '/' + fileName, 0)								#Leemos la imagen para mostrarla a continuación
		cv2.imshow('image', imagen)
		cv2.waitKey(10)

	label = label + 1

#cv2.destroyAllWindows()

face_recognizer = cv2.face.EigenFaceRecognizer_create()											#Especificamos el método a usar, en este caso EigenFaces

print("Entrenando...")
face_recognizer.train(facesData, np.array(labels))												#Entrenamos el programa con las imagenes previamente leídas

print("Creando modelo...")
face_recognizer.write('Modelos/modelo.xml')														#Almacenamos el modelo obtenido del entrenamiento
