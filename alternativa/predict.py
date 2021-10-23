import numpy as np
from keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model

longitud, altura = 150, 150
modelo = './modelo/modelo.h5'
pesos_modelo = './modelo/pesos.h5'
cnn = load_model(modelo)
cnn.load_weights(pesos_modelo)

def predict(file):
  x = load_img(file, target_size=(longitud, altura))
  x = img_to_array(x)
  x = np.expand_dims(x, axis=0)
  array = cnn.predict(x)
  result = array[0]
  answer = np.argmax(result)
  if answer == 0:
    print("pred: 1")
  elif answer == 1:
    print("pred: 2")
  elif answer == 2:
    print("pred: 3")
  elif answer == 3:
    print("pred: 4")
  elif answer == 4:
    print("pred: 5")
  elif answer == 5:
    print("pred: 6")  

  return answer

predict("./tmp/train/4/da__0_139736.png")
