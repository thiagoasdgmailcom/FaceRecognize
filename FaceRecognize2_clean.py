#!/usr/bin/env python
# coding: utf-8

# In[2]:


# Deep Learning CNN model para reconhecimento de face

# Local onde os arquivos de imagem para treino estão.
g_master_path = "D:\\GoogleDrive\\UFRJ\\EngenhariaDeSoftware\\IA\\MachineLearning\\FaceRecognize\\deep_learning"
g_face_training_path = 'final_training_images'
g_face_test_path = 'final_test_images'
g_face_live_path = 'final_live_images'

# biblioteca para trabalhar as imagens de treinamento
from keras.preprocessing.image import ImageDataGenerator
# https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html

# pré-processamento em imagens brutas de dados de treinamento
# Esses hiperparâmetros ajudam a gerar versões levemente distorcidas
# da imagem original, o que leva a um modelo melhor, pois aprende
# na mistura boa e ruim de imagens
train_datagen = ImageDataGenerator(
        # aplica distorcões na imagem.
        shear_range=0.1,
        # aplica zoom na imagem.
        zoom_range=0.1,
        # aplica espelhamento na imagem.
        horizontal_flip=True)

# Não é necessario fazer o pré-processamento em imagens brutas de dados de teste
test_datagen = ImageDataGenerator()

TrainingImagePath = g_master_path + '\\' + g_face_training_path

# Gerando dataset de treinamento
training_set = train_datagen.flow_from_directory(
        TrainingImagePath,
        target_size=(64, 64),
        batch_size=32,
        class_mode='categorical')

# Gerando dataset de teste
test_set = test_datagen.flow_from_directory(
        TrainingImagePath,
        target_size=(64, 64),
        batch_size=32,
        class_mode='categorical')

STEP_SIZE_TRAIN=training_set.n
STEP_SIZE_VALID=test_set.n

# Imprimindo labels para cada face
test_set.class_indices

# class_indices contem os indices para cada face
TrainClasses = training_set.class_indices

# Salva a face e o indice para futura referencia
ResultMap={}
for faceValue,faceName in zip(TrainClasses.values(),TrainClasses.keys()):
    ResultMap[faceValue]=faceName

# Salva o mapa de referencia para futura referencia
import pickle
with open("ResultsMap.pkl", 'wb') as fileWriteStream:
    pickle.dump(ResultMap, fileWriteStream)
    
print("Mapping of Face and its ID",ResultMap)

# Numero de neuronios de saida é igual ao numero de faces para identificar.
OutputNeurons=len(ResultMap)

print('\n The Number of output neurons: ', OutputNeurons)

# modelo deep learning CNN.
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPool2D
from keras.layers import Flatten
from keras.layers import Dense

# Inicializando a Rede Neural Convolucional
classifier= Sequential()

# Passo 1 Convolução
# Adicionando a primeira camada da CNN
# estamos usando o formato (64,64,3) porque estamos usando o backend do TensorFlow
# Significa 3 matrizes de tamanho (64X64) pixels representando os componentes vermelho, verde e azul dos pixels
classifier.add(Convolution2D(32, kernel_size=(5, 5), strides=(1, 1), input_shape=(64,64,3), activation='relu'))

# Passo 2 MAX Pooling
# Max Pooling é uma operação de agrupamento que calcula o valor máximo para patches de um mapa de recursos e # utiliza para criar um mapa de recursos reduzido (em pool). Geralmente é usado após uma camada
# convolucional. Ele adiciona uma pequena quantidade de invariância de tradução - o que significa que
# traduzir a imagem em uma pequena quantidade não afeta significativamente os valores da maioria das saídas agrupadas.
classifier.add(MaxPool2D(pool_size=(2,2)))

# Adicionando uma camada extra para melhor acuracia.
classifier.add(Convolution2D(64, kernel_size=(5, 5), strides=(1, 1), activation='relu'))
classifier.add(MaxPool2D(pool_size=(2,2)))

# Passo 3 FLattening
# A camada Flatten é usada para transformar os dados de entrada em um vetor unidimensional, 
# que é então usado como entrada para a camada Dense.
# Essa camada é usada porque a camada Dense requer uma entrada unidimensional, enquanto as camadas
# anteriores produzem saídas bidimensionais ou tridimensionais.
classifier.add(Flatten())

# Passo 4 Rede neural conectada
classifier.add(Dense(64, activation='relu'))

classifier.add(Dense(OutputNeurons, activation='softmax'))

# Compilando a CNN
classifier.compile(loss='categorical_crossentropy', optimizer = 'adam', metrics=["accuracy"])

import numpy as np

import time
# Mensurando tempo de treinamento
StartTime=time.time()

BATCH_SIZE = 32
# aqui é necessario colocar o numero de imagens no treinamento, obtido no comando anterior

TRAIN_STEPS_PER_EPOCH = np.ceil((STEP_SIZE_TRAIN*0.8/BATCH_SIZE)-1)
# para garantir que haja imagens suficientes para treinar bahch
VAL_STEPS_PER_EPOCH = np.ceil((STEP_SIZE_VALID*0.2/BATCH_SIZE)-1)

# Iniciando treinamento
classifier.fit(
                    training_set,
                    steps_per_epoch=TRAIN_STEPS_PER_EPOCH,
                    epochs=10,
                    validation_data=test_set,
                    validation_steps=VAL_STEPS_PER_EPOCH
)
EndTime=time.time()

print("###### Total Time Taken: ", round((EndTime-StartTime)/60), 'Minutes ######')



# In[3]:


# reconhecimento com live.

# Local onde os arquivos de imagem para treino estão.
g_master_path = "D:\\GoogleDrive\\UFRJ\\EngenhariaDeSoftware\\IA\\MachineLearning\\FaceRecognize\\deep_learning"
g_face_training_path = 'final_training_images'
g_face_test_path = 'final_test_images'
g_face_live_path = 'final_live_images'

# Biblioteca para criação das pastas onde os arquivos serão salvos.
import os

t_is_dir_exist = os.path.exists(g_master_path + '\\' + g_face_test_path)
if not t_is_dir_exist:
    # Criando diretorio do treinamento, caso não exista.
    os.makedirs(g_master_path + '\\' + g_face_test_path)
    print("The new directory is created: " + g_master_path + '\\' + g_face_test_path)
    
import cv2

import os

import numpy as np
import keras.utils as image

import time
from itertools import count
from multiprocessing import Process

def checkface():
    print('Starting function checkface()...')
    while True:
        time.sleep(5)
        print(next(checkface_counter))

# Capturando camera.
cap = cv2.VideoCapture(0)
print(g_master_path + "\\haarcascade_frontalface_default.xml")
# Abrindo arquivo pre definido (criado com machine learning) para detectar uma face em uma imagem.
faceCascade = cv2.CascadeClassifier(g_master_path + "\\haarcascade_frontalface_default.xml")
# faceCascade = cv2.CascadeClassifier(g_master_path + "\\lbpcascades\\lbpcascade_frontalface.xml")
# faceCascade = cv2.CascadeClassifier(g_master_path + "\\haarcascades\\haarcascade_eye.xml")
# faceCascade = cv2.CascadeClassifier(g_master_path + "\\haarcascades\\haarcascade_profileface.xml")
# --faceCascade = cv2.CascadeClassifier(g_master_path + "\\lbpcascades\\lbpcascade_profileface.xml")
# --faceCascade = cv2.CascadeClassifier(g_master_path + "\\lbpcascades\\lbpcascade_silverware.xml")

# Definindo count de capturas para nomear os arquivos resultado da captura da face.
img_count = 0

t_last_name = ""
t_last_names = ['']

t_checkface = False

# checkface_process = Process(target=checkface, name='Process_checkface')

# counter is an infinite iterator
# checkface_counter = count(0)

# checkface_process.start()

# checkface_process.join(timeout=5)

checkface_start_time=time.time()

while(True):
    # Captura frame-by-frame da camera
    ret, frame = cap.read()

    # transforma a iamgem do frame da camera em preto e branco
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detecta as faces na imagem
    faces = faceCascade.detectMultiScale(
        gray,
        # scaleFactor=1.1,
        # escala das faces, pois podem existir faces mais perto e mais longe do ponto de visão da camera.
        # minNeighbors=5,
        # tamanho minimo da face para detecção
        # minSize=(100, 100),
        # tamanho maximo da face para detecção
       #  maxSize=(300, 300)
    )

    count_faces = 0
    # Desenha um retangulo ao redor das faces encontradas nas imagens
    for (x, y, w, h) in faces:
        
        count_faces += 1
        
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        checkface_end_time=time.time()
        
        checkface_time=checkface_end_time-checkface_start_time

        # Verifica se a tecla “p” foi pressionada
        if cv2.waitKey(1) & 0xFF == ord('p'):
        # if checkface_time > 1:
            checkface_start_time=time.time()
            
            # recorta somente a imagem da face e salva na pasta de treinamento.
            crop_img = frame[y:y+h, x:x+w]
            cv2.imshow("cropped", crop_img)
            # Filename
            filename = g_master_path + '\\' + g_face_test_path + '\\face_' + str(img_count) + '.jpg'
            print(filename)
            img_count = img_count + 1
            # salvando
            cv2.imwrite(filename, crop_img)
   
            # Comparando imagem salva com banco de dados
            ImagePath=filename
            test_image=image.load_img(ImagePath,target_size=(64, 64))
            test_image=image.img_to_array(test_image)

            test_image=np.expand_dims(test_image,axis=0)

            result=classifier.predict(test_image,verbose=0)
            #print(training_set.class_indices)

            t_last_name = ResultMap[np.argmax(result)]
        
            t_last_names[count_faces - 1] = t_last_name
            
            print('####'*10)
            print(np.argmax(result))
            print(result)
            print('Prediction is: ', t_last_name[count_faces - 1])
            

        # Write some Text

        # font
        font = cv2.FONT_HERSHEY_SIMPLEX

        # org
        org = (x, y+h+20)

        # fontScale
        fontScale = 0.5

        # Blue color in BGR
        color = (0, 255, 0)

        # Line thickness of 2 pxq
        thickness = 1

        # Using cv2.putText() method
        frame = cv2.putText(frame, t_last_name, org, font, 
                           fontScale, color, thickness, cv2.LINE_AA)
    
    # Mostra a imagem com a face detectada
    cv2.imshow('frame', frame)

    # Para finalizar o script e sair da janela, aperto a tecla “q”
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Saindo...")
        break

# When everything done, release the capture
print("release...")
# checkface_process.terminate()
cap.release()
print("destroyAllWindows...")
cv2.destroyAllWindows()
print("Fim...")

