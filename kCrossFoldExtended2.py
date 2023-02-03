import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, StratifiedKFold
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, BatchNormalization, Dropout
from keras.models import Sequential
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from sklearn.metrics import confusion_matrix
from keras.applications import ResNet152V2

# setting main path

master_path = "data_eye"

train_path = os.path.join(master_path, "train")
test_path = os.path.join(master_path, "test")

###########################################################################################
# Add test data randomly from the train data sets

import random
import shutil

# source
open_train_path = os.path.join(train_path, "open_eye")
close_train_path = os.path.join(train_path, "closed_eye")

# destination
open_test_path = os.path.join(test_path, "open_eye")
close_test_path = os.path.join(test_path, "closed_eye")

source1 = open_train_path
destination1 = open_test_path
files1 = os.listdir(source1)
no_of_files1 = len(files1) // 5

for file_name1 in random.sample(files1, no_of_files1):
    shutil.copy(os.path.join(source1, file_name1), destination1)

source2 = close_train_path
destination2 = close_test_path
files2 = os.listdir(source2)
no_of_files2 = len(files2) // 5

# print(no_of_files1, no_of_files2)

for file_name2 in random.sample(files2, no_of_files2):
    shutil.copy(os.path.join(source2, file_name2), destination2)
###########################################################################################

open_train_images = glob.glob(train_path+"\\open_eye/*.png")
closed_train_images = glob.glob(train_path+"\\closed_eye/*.png")

open_test_images = glob.glob(test_path+"\\open_eye/*.png")
closed_test_images = glob.glob(test_path+"\\closed_eye/*.png")

train_list = [x for x in closed_train_images]
train_list.extend([x for x in open_train_images])

df_train = pd.DataFrame(np.concatenate([["closed_eye"]*len(closed_train_images),
                                        ["open_eye"]*len(open_train_images)]), columns=["class"])
df_train["image"] = [x for x in train_list]

test_list = [x for x in closed_test_images]
test_list.extend([x for x in open_test_images])

df_test = pd.DataFrame(np.concatenate([["closed_eye"]*len(closed_test_images),
                                       ["open_eye"]*len(open_test_images)]), columns=["class"])
df_test["image"] = [x for x in test_list]


# train_df, val_df = train_test_split(df_train, test_size=0.2, random_state=13, stratify=df_train["class"])
# Implement kFold split
# Storing the average of all predictions
main_pred = []
data_kfold = pd.DataFrame()

# Creating X, Y for training

train_y = df_train['class']

train_x = df_train.drop(['class'],axis=1)

# print(train_x)

# Define per-fold score containers <-- these are new
acc_per_fold = []
loss_per_fold = []

k=5
j=1
kfold = StratifiedKFold(n_splits=k,shuffle=True,random_state=None)
for train_df, val_df in list(kfold.split(train_x, train_y)):
    x_train_df = df_train.iloc[train_df]
    y_valid_df = df_train.iloc[val_df]
    train_datagen = ImageDataGenerator(rescale=1/255)
    val_datagen = ImageDataGenerator(rescale=1/255)
    train_generator = train_datagen.flow_from_dataframe(
                      x_train_df,
                      x_col="image",
                      y_col="class",
                      target_size=(150,150),
                      batch_size=32,
                      class_mode="binary",
                      seed=7)
    val_generator = val_datagen.flow_from_dataframe(
                    y_valid_df,
                    x_col="image",
                    y_col="class",
                    target_size=(150,150),
                    batch_size=32,
                    class_mode="binary",
                    seed=7)
    test_generator = val_datagen.flow_from_dataframe(
                     df_test,
                     x_col="image",
                     y_col="class",
                     target_size=(150,150),
                     batch_size=32,
                     class_mode="binary",
                     shuffle=False,
                     seed=7)
    #building the cnn
    model = Sequential()
    model.add(Conv2D(filters=16, kernel_size=(3,3), activation="relu", input_shape=(150,150,3)))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(filters=32, kernel_size=(3,3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(filters=64, kernel_size=(3,3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Flatten())
    model.add(Dense(64, activation="relu"))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    # Generate a print
    print('------------------------------------------------------------------------')
    print(f'Training for fold {j} ...')
    model_1 = model.fit(train_generator, epochs=10, validation_data=val_generator)

    ############################## Save Loss/Accuracy Figure ##################################
    plt.figure(figsize=(12,8))
    plt.subplot(2,2,1)
    plt.plot(model_1.history['loss'], label="Training Loss")
    plt.plot(model_1.history["val_loss"], label="Validation Loss")
    plt.legend()
    plt.title("Loss Evolution")

    plt.subplot(2,2,2)
    plt.plot(model_1.history['accuracy'], label="Training Loss")
    plt.plot(model_1.history["val_accuracy"], label="Validation Loss")
    plt.legend()
    plt.title("Accuracy Evolution")
    plt.savefig("img\histPlot" + str(j) + ".jpg")
    ###########################################################################################

    ############################## Save Confustion Matrix Figure ##############################
    from sklearn.metrics import confusion_matrix
    y_true = test_generator.classes
    y_pred = (model.predict(test_generator) > 0.5).astype("int32")

    plt.figure(figsize=(12,5))

    confusion_matrix = confusion_matrix(y_true, y_pred)
    sns.heatmap(confusion_matrix, annot=True, fmt="d")

    plt.xlabel("Predicted Label", fontsize=12)
    plt.xlabel("True Label", fontsize=12)

    plt.savefig("img\confustionMat" + str(j) + ".jpg")
    ###########################################################################################

    ############################## Save ROC/AUC Figure ########################################
    from sklearn.metrics import roc_curve,auc

    y_preds = model.predict(test_generator).ravel()

    fpr, tpr, thresholds = roc_curve(y_true, y_preds)
    roc_auc=auc(fpr,tpr)
    plt.plot(fpr,tpr,label='AUC=%0.4f'%roc_auc,color='darkorange')
    plt.legend(loc='lower right')
    plt.figure(figsize=(8,8))
    plt.plot([0,1], [0,1], "y--")
    plt.plot(fpr, tpr, marker=".")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("ROC curve")
    plt.savefig("img/rocCurve" + str(j) + ".jpg")
    ###########################################################################################
    
    # Generate generalization metrics
    scores = model.evaluate(test_generator)
    print(f'Score for fold {j}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1]*100}%')
    acc_per_fold.append(scores[1] * 100)
    loss_per_fold.append(scores[0])
    j+=1

# == Provide average scores ==
print('------------------------------------------------------------------------')
print('Score per fold')
for i in range(0, len(acc_per_fold)):
  print('------------------------------------------------------------------------')
  print(f'> Fold {i+1} - Loss: {loss_per_fold[i]} - Accuracy: {acc_per_fold[i]}%')
print('------------------------------------------------------------------------')
print('Average scores for all folds:')
print(f'> Accuracy: {np.mean(acc_per_fold)} (+- {np.std(acc_per_fold)})')
print(f'> Loss: {np.mean(loss_per_fold)}')
print('------------------------------------------------------------------------')
print('------------------------------------------------------------------------')
print('---------------------- Hyperparameter Tuning ---------------------------')


# == Apply image augmentation ==  #
# Do the k-cross validation again #
j=1
for train_df, val_df in list(kfold.split(train_x, train_y)):
    x_train_df = df_train.iloc[train_df]
    y_valid_df = df_train.iloc[val_df]
    train_datagen_2 = ImageDataGenerator(
                      rescale=1./255,
                      shear_range=0.2,
                      zoom_range=0.2,
                      horizontal_flip=True,
                      rotation_range=10,
                      fill_mode="nearest")
    val_datagen_2 = ImageDataGenerator(rescale=1./255)
    train_generator_2 = train_datagen_2.flow_from_dataframe(
                        x_train_df,
                        x_col="image",
                        y_col="class",
                        target_size=(150,150),
                        batch_size=32,
                        class_mode="binary",
                        seed=7)
    val_generator_2 = val_datagen_2.flow_from_dataframe(
                      y_valid_df,
                      x_col="image",
                      y_col="class",
                      target_size=(150,150),
                      batch_size=32,
                      class_mode="binary",
                      seed=7)
    test_generator_2 = val_datagen_2.flow_from_dataframe(
                       df_test,
                       x_col="image",
                       y_col="class",
                       target_size=(150,150),
                       batch_size=32,
                       class_mode="binary",
                       shuffle=False,
                       seed=7)

    reduce_lr = ReduceLROnPlateau(monitor="val_accuracy", factor=0.1, min_delta=0.0001, patience=1, verbose=1)
    filepath=("weights\weights" + str(j) + ".hdf5")
    checkpoint = ModelCheckpoint(filepath, monitor="val_accuracy", verbose=1, save_best_only=True, mode="max")

    # Create the cnn model with tuned hyperparameter 
    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=(3,3), activation="relu", padding="same", input_shape=(150, 150, 3)))

    model.add(Conv2D(filters=32, kernel_size=(3,3), activation="relu", padding="same"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Conv2D(filters=64, kernel_size=(3,3), activation="relu", padding="same"))
    model.add(Conv2D(filters=64, kernel_size=(3,3), activation="relu", padding="same"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Flatten())
    model.add(Dense(128, activation="relu"))
    model.add(BatchNormalization())
    model.add(Dropout(rate=0.2))
    model.add(Dense(1, activation="sigmoid"))

    model.compile(loss="binary_crossentropy", optimizer=Adam(learning_rate=0.0001), metrics=["accuracy"])
    print('------------------------------------------------------------------------')
    print(f'Training for fold {j} ...')
    # Train the model
    htuning_model = model.fit(
                    train_generator_2,
                    epochs=10,
                    validation_data=val_generator,
                    callbacks=[reduce_lr, checkpoint])
    
    ############################## Save Loss/Accuracy Figure ##################################
    plt.figure(figsize=(12,8))
    plt.subplot(2,2,1)
    plt.plot(htuning_model.history['loss'], label="Training Loss")
    plt.plot(htuning_model.history["val_loss"], label="Validation Loss")
    plt.legend()
    plt.title("Loss Evolution")

    plt.subplot(2,2,2)
    plt.plot(htuning_model.history['accuracy'], label="Training Loss")
    plt.plot(htuning_model.history["val_accuracy"], label="Validation Loss")
    plt.legend()
    plt.title("Accuracy Evolution")
    plt.savefig("hyperPimg\histPlot" + str(j) + ".jpg")
    ###########################################################################################

    ############################## Save Confustion Matrix Figure ##############################
    from sklearn.metrics import confusion_matrix
    y_true = test_generator_2.classes
    y_pred = (model.predict(test_generator_2) > 0.5).astype("int32")    

    plt.figure(figsize=(12,5))

    confusion_matrix = confusion_matrix(y_true, y_pred)
    sns.heatmap(confusion_matrix, annot=True, fmt="d")

    plt.xlabel("Predicted Label", fontsize=12)
    plt.xlabel("True Label", fontsize=12)

    plt.savefig("hyperPimg\confustionMat" + str(j) + ".jpg")
    ###########################################################################################

    ############################## Save ROC/AUC Figure ########################################
    from sklearn.metrics import roc_curve,auc

    y_preds = model.predict(test_generator_2).ravel()

    fpr, tpr, thresholds = roc_curve(y_true, y_preds)
    roc_auc=auc(fpr,tpr)
    plt.plot(fpr,tpr,label='AUC=%0.4f'%roc_auc,color='darkorange')
    plt.legend(loc='lower right')
    plt.figure(figsize=(8,8))
    plt.plot([0,1], [0,1], "y--")
    plt.plot(fpr, tpr, marker=".")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("ROC curve")
    plt.savefig("hyperPimg/rocCurve" + str(j) + ".jpg")
    ###########################################################################################

    # Generate generalization metrics
    scores = model.evaluate(test_generator_2)
    print(f'Score for fold {j}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1]*100}%')
    acc_per_fold.append(scores[1] * 100)
    loss_per_fold.append(scores[0])
    j+=1

# == Provide average scores ==
print('------------------------------------------------------------------------')
print('Score per fold')
for i in range(0, len(acc_per_fold)):
  print('------------------------------------------------------------------------')
  print(f'> Fold {i+1} - Loss: {loss_per_fold[i]} - Accuracy: {acc_per_fold[i]}%')
print('------------------------------------------------------------------------')
print('Average scores for all folds:')
print(f'> Accuracy: {np.mean(acc_per_fold)} (+- {np.std(acc_per_fold)})')
print(f'> Loss: {np.mean(loss_per_fold)}')
print('------------------------------------------------------------------------')

