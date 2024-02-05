#
#Autor: Roberto Gijón Liétor
#
#Descripción: Este programa sirve para capturar y almacenar las caras de una persona através de un vídeo.
#


import cv2
import os
import imutils

nombre = 'Nombre'																								#Nombre de la persona que se desea reconocer
directorioDatos = '.../Datos'						                                                            #Directorio donde se va a almacenar la carpeta de cada persona
directotioPersona = directorioDatos + '/' + nombre 																#Directorio donde se almacenará las capturas de la cara de la persona quue vayamos a añadir
	
if not os.path.exists(directotioPersona):																		#Si no existe la carpeta, la creamos
	os.makedirs(directotioPersona)
	print('Directorio de ' + nombre + ' creado en: "' + directotioPersona + '".')

cap = cv2.VideoCapture('Videos/video.mp4')							    										#Almacenamos el video desde donde vamos a hacer las capturas

faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')				#Inicializamos el detector de caras
cont = 0     

while True:
	ret, frame = cap.read()																						#Leemos cada fotograma del video
	if ret == False: break
	frame = imutils.resize(frame, width=640)																	#Redimensionamos para que todos los frames de todos los videos sean iguales
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)																#Pasamos cada fotograma a escala de grises
	auxFrame = frame.copy()

	faces = faceClassif.detectMultiScale(gray,1.3,5)															#Obtenemos un vector con las coordenadas de la cara

	for (x,y,w,h) in faces:																						#Recorremos el vector
		cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)														#Hacemos un cuadrado a la cara, con las coordenadas del vector
		cara = auxFrame[y:y+h,x:x+w]
		cara = cv2.resize(cara,(150,150),interpolation=cv2.INTER_CUBIC)											#Ajustamos cada captura de cara con el mismo tamaño, 150x150
		cv2.imwrite(directotioPersona + '/cara_{}.jpg'.format(cont),cara)										#Guardamos cada imagen
		cont += 1

	cv2.imshow('frame', frame)																					#Mostramos el proceso

	k = cv2.waitKey(1)
	if k == 27 or cont >= 300:																					#Hacemos esto 300 veces para obtener capturas de la cara de la persona del video (o hasta que pulsemos la tecla que corresponde a la tecla 27, ESC en este caso)
		break

cap.release()
cv2.destroyAllWindows()
