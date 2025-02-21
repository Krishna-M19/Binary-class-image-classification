################## TODO #####################
# install tensorflow, pandas, numpy, PIL
# Import libraries
import os
import pandas as pd
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import shutil
import glob



# Define the source directory containing the images
source_directory = "train/"

# Ensure the existence of the cat and dog folders
cat_folder = os.path.join(source_directory, "cat")
dog_folder = os.path.join(source_directory, "dog")

os.makedirs(cat_folder, exist_ok=True)
os.makedirs(dog_folder, exist_ok=True)

# Get a list of files in the source directory
files = os.listdir(source_directory)

# Filter files starting with "cat." or "dog."
cat_files = [file for file in files if file.startswith("cat.")]
dog_files = [file for file in files if file.startswith("dog.")]

# Move cat images to the cat folder

# cat_files= cat_files[:10]
# dog_files= dog_files[:10]

for cat_file in cat_files:
    src = os.path.join(source_directory, cat_file)
    dst = os.path.join(cat_folder, cat_file)
    shutil.move(src, dst)
    print(f"Moved {cat_file} to cat folder.")


# Move dog images to the dog folder
for dog_file in dog_files:
      src = os.path.join(source_directory, dog_file)
      dst = os.path.join(dog_folder, dog_file)
      shutil.move(src, dst)
      print(f"Moved {dog_file} to dog folder.")

# Count the number of train sample
train_files = glob.glob('train/*')
len_train = len(train_files)

def prepare_val_train_dataset(train_data_dir = 'train/'):
    # Define the input shape and batch size
    input_shape = (128, 128, 3)
    batch_size = 32
    # Create an ImageDataGenerator for data augmentation and preprocessing
    data_generator = ImageDataGenerator(
        rescale = 1.0 / 255.0,  # Rescale pixel values to [0, 1]
        validation_split = 0.2,  # 20% of the data will be used for validation
        )

    catago =['cat','dog']

    # Load and preprocess the training data
    train_data = data_generator.flow_from_directory(
        train_data_dir,
        target_size=input_shape[:2],
        batch_size=batch_size,
        class_mode='binary', # Binary classification
        subset='training', # Training split
        classes =catago
        )
    # Load and preprocess the validation data
    validation_data = data_generator.flow_from_directory(
        train_data_dir,
        target_size=input_shape[:2],
        batch_size=batch_size,
        class_mode='binary', # Binary classification
        subset='validation', # Validation split
        classes =catago
        )
    ########################### TODO:A ###########################
    # Observe shapes of validation and training dataset


    height, width, channels = input_shape

    # Shape and dimensions of training data
    train_data_shape = (train_data.samples, height, width, channels)

    # Shape and dimensions of validation data
    validation_data_shape = (validation_data.samples, height, width, channels)

    print("Shape and dimensions of training data:", train_data_shape)
    print("Shape and dimensions of validation data:", validation_data_shape)
    #################### YOUR CODE ENDS HERE #####################
    # Create a dictionary to
    class_to_label = train_data.class_indices
    # print classes to labels mapping (cat: 0, dog: 1)
    print("Class to Label Mapping:", class_to_label)
    return train_data, validation_data



def prepare_test_dataset(test_data_dir = 'test1/'):
    test_data =[]
    for img_file in os.listdir(test_data_dir):
        img_path = os.path.join(test_data_dir, img_file)
        image = Image.open(img_path)
        image = image.resize((128, 128))
        image_array = np.array(image)
        image_array = image_array / 255.0 # Normalize pixel values to [0, 1]
        image = np.expand_dims(image_array, axis=0)
        test_data.append(image)
    return test_data


test_data = prepare_test_dataset('test1/')


from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

class DogVsCatClassifier:
    def __init__(self, input_shape=(128, 128, 3)):
        self.model = self.build_model(input_shape)
        self.compile_model()

    def build_model(self, input_shape):
        model = keras.Sequential()
        # Convolutional layers
        model.add(layers.Conv2D(32, (3, 3), activation='relu',padding='same',input_shape=input_shape))
        # model.add(layers.Conv2D(32, (3, 3), activation='relu',padding='same'))
        model.add(layers.MaxPooling2D((2, 2), strides=(2, 2)))


        model.add(layers.Conv2D(64, (3, 3), activation='relu',padding='same'))
        # model.add(layers.Conv2D(64, (3, 3), activation='relu',padding='same'))
        model.add(layers.MaxPooling2D((2, 2), strides=(2, 2)))


        model.add(layers.Conv2D(128, (3, 3), activation='relu',padding='same'))
        model.add(layers.Conv2D(128, (3, 3), activation='relu',padding='same'))
        model.add(layers.MaxPooling2D((2, 2), strides=(2, 2)))


        # Flatten the output from convolutional layers
        model.add(layers.Flatten())
        # Fully connected layers
        model.add(layers.Dense(256, activation='relu')) # was 512
        model.add(layers.Dropout(0.5))  # Adding a dropout layer
        model.add(layers.Dense(256, activation='relu')) # was 512
        # model.add(layers.Dropout(0.5))  # Adding a dropout layer
        model.add(layers.Dense(1, activation='sigmoid')) # Binary classification

        return model

    def compile_model(self):
        self.model.compile(
        # optimizer='adam',
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),

        loss='binary_crossentropy',
        metrics=['accuracy']
        )

    def train(self, train_data, validation_data, epochs=30, batch_size=32):
        history = self.model.fit(
        train_data,
        epochs=epochs,
        validation_data=validation_data,
        #batch_size=batch_size
        )
        return history
    def evaluate(self, validation_data ,true_labels):
        #### TASK C: Expand to calculate Sensitivity, Specificity, F1 Score
        accuracy =self.model.evaluate(validation_data)[1]

        #  Make predictions on the test set
        y_pred = self.model.predict(validation_data)
        y_pred_binary = (y_pred > 0.5).astype('int32')

        # print("y_pred:",y_pred)
        # print("y_pred_binary: " , y_pred_binary)
        # print("true_labels: ",true_labels)
        # precision = precision_score(true_labels, y_pred_binary)
        # recall = recall_score(true_labels, y_pred_binary)
        f_one = f1_score(true_labels, y_pred_binary)

        # Calculate confusion matrix
        conf_matrix = confusion_matrix(true_labels, y_pred_binary)

        # Extract true positives, true negatives, false positives, and false negatives
        tn, fp, fn, tp = conf_matrix.ravel()

        # Calculate sensitivity (True Positive Rate)
        sensitivity = tp / (tp + fn)
        # Calculate specificity (True Negative Rate)
        specificity = tn / (tn + fp)
        return accuracy ,sensitivity, specificity,f_one

    def predict(self, image):
        # Assuming you have preprocessed the input image
        return self.model.predict(image)

    def save_model(self, model_path):
        self.model.save(model_path)

    def load_model(self, model_path):
        self.model = keras.models.load_model(model_path)

if __name__ == '__main__':
    # Prepare dataset
    train_data, validation_data = prepare_val_train_dataset('train/')

    validation_labels = validation_data.classes
    # print("Validation labels:", validation_labels)
    class_to_index =validation_data.class_indices
    # Reverse the mapping to get index to class mapping
    index_to_class = {v: k for k, v in class_to_index.items()}

    # Map the labels back to class names
    validation_class_names = [index_to_class[label] for label in validation_labels]
    # print("Validation class names:", validation_class_names)



    # create classifier instance
    classifier = DogVsCatClassifier()
    # Train the model using your training data and validation data
    classifier.train(train_data, validation_data)
    # Evaluate the model on validation dataset

    accuracy,sensitivity ,specificity ,f_one = classifier.evaluate(validation_data,validation_labels)

    print(f"Sensitivity: {sensitivity:.4f}")
    print(f"Specificity: {specificity:.4f}")
    print(f"F1 Score: {f_one:.4f}")
    print(f"Accuracy: {accuracy:.4f}")


    # save model

    classifier.save_model('/content/drive/MyDrive/Artificial intelligence(cs4811)/model')
    ###### TASK D: Expand the below given code to test all the images in test1 folder. ######
    # Create a CSV file with column names:
    # Image name, probability_score, predicted_class for all 12500 images
    # Load a test image (example: 'test_image.jpg')

    # image = Image.open('test1/1.jpg')
    # # Preprocess the input image to match the model's input shape (128x128 pixels
    # # with 3 color channels)
    # image = image.resize((128, 128))
    # image_array = np.array(image)
    # image_array = image_array / 255.0 # Normalize pixel values to [0, 1]

    prediction_list = []
    prediction_class =[]
    # # Make a prediction on the preprocessed image
    for i in range(len(test_data)):
        # for i in range(10):
        pred = classifier.predict(test_data[i])
        prediction_list.append(pred)

    if pred[0][0] > 0.5:
        prediction_class.append(1)
    else :
        prediction_class.append(0)

    print("predictions",prediction_list)
    print("single",prediction_list[0][0][0])
    # # The 'prediction' variable contains the model's output, which is a probability
    # # You can interpret the result based on your class labels
    # if prediction[0][0] > 0.5:
    #   print("It's a dog!")
    # else:
    #   print("It's a cat!")
    image_names = os.listdir('test1/')
    # print(len(image_names[:10]) ,image_names)
    df = pd.DataFrame({'Image name':image_names, 'probability_score':prediction_list, 'predicted_class':prediction_class})
    df.to_csv('predictions.csv', index=False)