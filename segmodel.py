#Main AI libraries
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
#TFDS for easy dataset access
import tensorflow_datasets as tfds
#For visualization
import matplotlib.pyplot as plt
#for Array operations
import numpy as np
#For fs access
import os
#Simple image manupilation
from PIL import Image
#Web UI access
import webbrowser

#Url of the web application and the chrome application path 
url = 'http://localhost:8000'
chrome_path = 'C:/Program Files (x86)/Google/Chrome/Application/chrome.exe %s'

#Getting Dataset
dataset, info = tfds.load('oxford_iiit_pet:3.*.*', with_info=True)

#Function to resize images to 128x128
def resize(input_image, input_mask):
   input_image = tf.image.resize(input_image, (128, 128), method="nearest")
   input_mask = tf.image.resize(input_mask, (128, 128), method="nearest")
   return input_image, input_mask


#Augmentation function for training
def augment(input_image, input_mask):
   if tf.random.uniform(()) > 0.5:
       # Random flipping of the image and mask
       input_image = tf.image.flip_left_right(input_image)
       input_mask = tf.image.flip_left_right(input_mask)
   return input_image, input_mask

#Normalizing input images Scaling pixels in the range 0-1 
def normalize(input_image, input_mask):
   input_image = tf.cast(input_image, tf.float32) / 255.0
   input_mask -= 1
   return input_image, input_mask

#Normalizing input images
def normalize2(input_image):
   input_image = tf.cast(input_image, tf.float32) / 255.0
   return input_image


#Preparing train and testing dataset parts
def load_image_train(datapoint):
   input_image = datapoint["image"]
   input_mask = datapoint["segmentation_mask"]
   input_image, input_mask = resize(input_image, input_mask)
   input_image, input_mask = augment(input_image, input_mask)
   input_image, input_mask = normalize(input_image, input_mask)
   return input_image, input_mask
def load_image_test(datapoint):
   input_image = datapoint["image"]
   input_mask = datapoint["segmentation_mask"]
   input_image, input_mask = resize(input_image, input_mask)
   input_image, input_mask = normalize(input_image, input_mask)
   return input_image, input_mask

#Applying functions to the dataset
train_dataset = dataset["train"].map(load_image_train, num_parallel_calls=tf.data.AUTOTUNE)
test_dataset = dataset["test"].map(load_image_test, num_parallel_calls=tf.data.AUTOTUNE)

#Defining batch and buffer size and creating batches 64 gives out of memory
BATCH_SIZE = 25
BUFFER_SIZE = 1000
#Filling buffer with random images
train_batches = train_dataset.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()
#Prefectch to decrease latency
train_batches = train_batches.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
validation_batches = test_dataset.take(3000).batch(BATCH_SIZE)
test_batches = test_dataset.skip(3000).take(669).batch(BATCH_SIZE)

#Function to images on the screen
def display(display_list):
 plt.figure(figsize=(15, 15))
 title = ["Input Image", "Predicted Mask","True Mask"]
 for i in range(len(display_list)):
   plt.subplot(1, len(display_list), i+1)
   plt.title(title[i])
   plt.imshow(tf.keras.utils.array_to_img(display_list[i]))
   plt.axis("off")
 plt.show()




#Getting a random image/mask pair from test dataset
sample_batch = next(iter(test_batches))
random_index = np.random.choice(sample_batch[0].shape[0])
sample_image, sample_mask = sample_batch[0][random_index], sample_batch[1][random_index]
#display([sample_image, sample_mask])


#Starting model definition

#Repeated convolution block
def double_conv_block(x, n_filters):
   # Conv2D then ReLU activation
   x = layers.Conv2D(n_filters, 3, padding = "same", activation = "relu", kernel_initializer = "he_normal")(x)
   # Conv2D then ReLU activation
   x = layers.Conv2D(n_filters, 3, padding = "same", activation = "relu", kernel_initializer = "he_normal")(x)
   return x

#Downsampling block with dropout to avoid overfitting
def downsample_block(x, n_filters):
   f = double_conv_block(x, n_filters)
   p = layers.MaxPool2D(2)(f)
   p = layers.Dropout(0.3)(p)
   return f, p

#Upsampling block with concatenation last step in the model
def upsample_block(x, conv_features, n_filters):
   # upsample
   x = layers.Conv2DTranspose(n_filters, 3, 2, padding="same")(x)
   # concatenate
   x = layers.concatenate([x, conv_features])
   # dropout
   x = layers.Dropout(0.3)(x)
   # Conv2D twice with ReLU activation
   x = double_conv_block(x, n_filters)
   return x


#Constructing the model
def build_unet_model():
  # inputs
   inputs = layers.Input(shape=(128,128,3))

   # encoder: contracting path - downsample
   # 1 - downsample
   f1, p1 = downsample_block(inputs, 64)
   # 2 - downsample
   f2, p2 = downsample_block(p1, 128)
   # 3 - downsample
   f3, p3 = downsample_block(p2, 256)
   # 4 - downsample
   f4, p4 = downsample_block(p3, 512)

   # 5 - bottleneck
   bottleneck = double_conv_block(p4, 1024)

   # decoder: expanding path - upsample
   # 6 - upsample
   u6 = upsample_block(bottleneck, f4, 512)
   # 7 - upsample
   u7 = upsample_block(u6, f3, 256)
   # 8 - upsample
   u8 = upsample_block(u7, f2, 128)
   # 9 - upsample
   u9 = upsample_block(u8, f1, 64)

   # outputs
   outputs = layers.Conv2D(3, 1, padding="same", activation = "softmax")(u9)

   # unet model with Keras Functional API
   unet_model = tf.keras.Model(inputs, outputs, name="U-Net")

   return unet_model
#Initializing model
unet_model = build_unet_model()

#Compiling model

#Lower batch count (20) and RMSprop lowered accuracy
# Categorical crossentropy for one-hot encoded
# sparse_categorical_crossentropy for integers
unet_model.compile(optimizer=tf.keras.optimizers.RMSprop(),
                  loss="sparse_categorical_crossentropy",
                  metrics="accuracy")

#Number of epochs to train
NUM_EPOCHS = 100

#Length of training set
TRAIN_LENGTH = info.splits["train"].num_examples
#Count of batches to train for one epoch
STEPS_PER_EPOCH = TRAIN_LENGTH // BATCH_SIZE

VAL_SUBSPLITS = 5
TEST_LENTH = info.splits["test"].num_examples
#Steps per epoch in validation size
VALIDATION_STEPS = TEST_LENTH // BATCH_SIZE // VAL_SUBSPLITS


#Creating checkpoint functionality
checkpoint_path = "training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

#Model Export path
""" save_path = "models/model"
save_dir = os.path.dirname(save_path) """
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

#Load latest saved weights
#Not uploading checkpoints because of the github filesize limit
try:
   latest = tf.train.latest_checkpoint(checkpoint_dir)
   unet_model.load_weights(latest)
except:
  print("Checkpoint Not Found!")


#Export model to a file
#unet_model.save(save_dir)
#Train function 
"""  model_history = unet_model.fit(train_batches,
                              epochs=NUM_EPOCHS,
                              steps_per_epoch=STEPS_PER_EPOCH,
                              validation_steps=VALIDATION_STEPS,
                              validation_data=validation_batches,
                              callbacks=[cp_callback])   """
                               

#Evaluate the model with test batches
#model_eval = unet_model.evaluate(test_batches,batch_size=BATCH_SIZE)


#Creating mask image
def create_mask(pred_mask):
#Gettng the biggest values to create mask
 pred_mask = tf.argmax(pred_mask, axis=-1)
 pred_mask = pred_mask[..., tf.newaxis]
 return pred_mask[0]

#Get predictions
#If filename exists get file predictions
#Else random from test dataset
def show_predictions(filename = False):
 global predimg
 global maskimg
 if filename:
    mask = create_mask(unet_model.predict(returnTensorIMG(filename)))
     #Preparing predicted mask and image to export
    predimg = Image.open(filename)
    maskimg = tf.keras.utils.array_to_img(mask)
    display([predimg,mask])
    
 else:
    mask = create_mask(unet_model.predict(sample_image[tf.newaxis, ...]))
    #Preparing predicted mask and image to export
    predimg = tf.keras.utils.array_to_img(sample_image)
    maskimg = tf.keras.utils.array_to_img(mask)
    display([sample_image,mask])
 

#For inputting image from outside
def returnTensorIMG(filename):
   testimage = Image.open(filename)
   testimage = testimage.resize((128,128))
   imgtensor = tf.convert_to_tensor(testimage, dtype=tf.float32)
   #Normalize image colors 0-1
   imgtensor = normalize2(imgtensor);
   return imgtensor[tf.newaxis,...]

#Save image to disk as rgba
def saveImage(image,name):
   image.convert('RGBA')
   image = image.resize((128,128))
   image.save(name,"png")


print("Length of train dataset",len(dataset["train"]),"Length of test dataset",len(dataset["test"]))

#Examples1 .jpeg: cat8,cat10,cat11
#Examples2 .jpeg: dog20,dog21,dog22

# Examples3 .jpeg :cat5 ,dog3 ,dog4,dog5,dog25
show_predictions("testimgs/cat8.jpeg");

# Saving images to the disk
saveImage(predimg,"predimg.png")
saveImage(maskimg,"mask.png")
#Starting browser for further processing
webbrowser.get(chrome_path).open(url)

