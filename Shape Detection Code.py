import os
import cv2
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.optimizers import SGD
from keras.models import Sequential
from keras.layers.core import Dense, Flatten, Activation, Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D, AveragePooling2D


IMG_SIZE=64

#create lists to save the labels (the name of the shape)
#labels, images, shape_dir = [],[], '../input/shapes/'
shapes = ['square', 'circle', 'triangle']
Shapes=shapes
images,labels=[],[]
#iterate through each shape
for shape in shapes:
    print('Getting data for: ', shape)
    #iterate through each file in the folder
    for image_name in os.listdir(shape):
        #add the image to the list of images
        image = cv2.imread(shape+'/'+image_name)
        image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
        images.append(image)
        #add an integer to the labels list 
        labels.append(shapes.index(shape))

print("\nDataset Images size:", len(images))
print("Image Shape:", images[0].shape)
print("Labels size:", len(labels))
Labels=labels
#break data into training and test sets
Dataset=images       
        
train_test_ratio, to_train = 5,0
train_images, test_images, train_labels, test_labels = [],[],[],[]


for image, label in zip(images, labels):
    if to_train<train_test_ratio: 
        train_images.append(image)
        train_labels.append(label)
        to_train+=1
    else:
        test_images.append(image)
        test_labels.append(label)
        to_train = 0

#could also use Sickit learn train test split
        


#overreview of the data
print('Number of training images: ', len(train_images))
print('Number of test images: ', len(test_images))

print("Count of Circles images:", Labels.count(Shapes.index("circle")))
print("Count of Squares images:", Labels.count(Shapes.index("square")))
print("Count of Triangle images:", Labels.count(Shapes.index("triangle")))





Dataset = np.array(Dataset)
Dataset = Dataset.astype("float32") / 255.0

# One hot encode labels
Labels = np.array(Labels)
Labels = to_categorical(Labels)

# Split Dataset to train\test
(trainX, testX, trainY, testY) = train_test_split(Dataset, Labels, test_size=0.2, random_state=42)

print("X Train shape:", trainX.shape)
print("X Test shape:", testX.shape)
print("Y Train shape:", trainY.shape)
print("Y Test shape:", testY.shape)

class LeNet():
    @staticmethod
    def build(numChannels, imgRows, imgCols, numClasses,  pooling= "max", activation= "relu"):
        # initialize the model
        model = Sequential()
        inputShape = (imgRows, imgCols, numChannels)

        # add first set of layers: Conv -> Activation -> Pool
        model.add(Conv2D(filters= 16, kernel_size= 5, input_shape= inputShape))
        model.add(Activation(activation))

        if pooling == "max":
            model.add(MaxPooling2D(pool_size= (2, 2), strides= (2, 2)))
        else:
            model.add(AveragePooling2D(pool_size= (2, 2), strides= (2, 2)))

        # add second set of layers: Conv -> Activation -> Pool
        model.add(Conv2D(filters= 32, kernel_size= 5))
        model.add(Activation(activation))

        if pooling == "avg":
            model.add(AveragePooling2D(pool_size=(2, 2), strides=(2, 2)))
        else:
            model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        # Flatten -> FC 120 -> Dropout -> Activation
        model.add(Flatten())
        model.add(Dense(120))
        model.add(Dropout(0.5))
        model.add(Activation(activation))

        # FC 84 -> Dropout -> Activation
        model.add(Dense(84))
        model.add(Dropout(0.5))
        model.add(Activation(activation))

        # FC 4-> Softmax
        model.add(Dense(numClasses))
        model.add(Activation("softmax"))

        return model
BS = 120
LR = 0.01
EPOCHS = 5
opt = SGD(lr= LR)

model = LeNet.build(3, IMG_SIZE, IMG_SIZE, 3, pooling= "max")
model.compile(loss= "categorical_crossentropy", optimizer= opt, metrics= ["accuracy"])
model.summary()

# Train model
H1 = model.fit(trainX, trainY, validation_data= (testX, testY), batch_size= BS,epochs= EPOCHS, verbose=1)

# Evaluate the train and test data
scores_train = model.evaluate(trainX, trainY, verbose= 1)
scores_test = model.evaluate(testX, testY, verbose= 1)

print("\nModel with Max Pool Accuracy on Train Data: %.2f%%" % (scores_train[1]*100))
print("Model with Max Pool Accuracy on Test Data: %.2f%%" % (scores_test[1]*100))

#model.save('shape_recg.h5')

































