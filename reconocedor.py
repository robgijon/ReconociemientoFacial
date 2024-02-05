#
#Autor: Roberto Gijón Liétor
#
#Descripción: Este programa sirve para reconocer caras tanto en video como por webCam, usando el modelo del entrenamiento anterior.
#


import cv2
import os

directorioDatos = 'C:/Users/rober/Desktop/Proyecto practico - Reconocimiento facial/Datos'						#Directorio de la carpeta de datos
nombres = os.listdir(directorioDatos)																			#Obtenemos todos los nombres de las personas almacenadas
print('Personas registeradas: ', nombres)

face_recognizer = cv2.face.EigenFaceRecognizer_create()

face_recognizer.read('Modelos/modelo.xml')																		#Leemos el modelo del entreamiento

cap = cv2.VideoCapture('Videos/video.mp4')		#Conocido													    #Almacenamos el video en el cual ejecutaremos el reconocimiento facial
#cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)			#Cámara														#Esta línea sería para hacerlo desde la webCam 


faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')				# \
																												# |
while True:																										# |
	ret,frame = cap.read()																						# |
	if ret == False:																							# |
		break																									# |	
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)																#  >  Mismo procedimiento que en el capturador de caras
	auxFrame = gray.copy()																						# |
																												# |		
	faces = faceClassif.detectMultiScale(gray, 1.3, 5)															# |		
																												# |		
	for(x,y,w,h) in faces:																						# |
		cara = auxFrame[y:y+h,x:x+w]																			# |
		cara = cv2.resize(cara,(150,150), interpolation=cv2.INTER_CUBIC)										# /
		result = face_recognizer.predict(cara)																	# Intenta predecir una etiqueta y un intervalo de confianza con la imagen que le mandamos como parámetro

		cv2.putText(frame,'{}'.format(result),(x,y-5),1,1.3,(255,255,0),1,cv2.LINE_AA)							#Mostramos la etiqueta y el intervalo obtenido 

		if result[1] < 5800:																					#Si el intervalo es menor a 5800 (no es un número fijo, pero probando es el que mejor me ha funcionado) es acertado
			cv2.putText(frame,'{}'.format(nombres[result[0]]),(x,y-25),2,1.1,(0,255,0),1,cv2.LINE_AA)			#Mostramos el nombre de la persona 
			cv2.rectangle(frame, (x,y),(x+w,y+h), (0,255,0),2)													#Dibujamos un cuadrado en la cara
		else:
			cv2.putText(frame,'Desconocido',(x,y-20),2,0.8,(0,0,255),1,cv2.LINE_AA)								#Si el intervalo está por encima de lo requerido, se mostrará como "Desconocido"
			cv2.rectangle(frame, (x,y),(x+w,y+h), (0,0,255),2)													


	cv2.imshow('frame', frame)
	k = cv2.waitKey(1)
	if k == 27:
		break

cap.release()
cv2.destroyAllWindows()
