#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020.2.28 23:38
# @Author  : Tony
# @Site    : 
# @File    : train_emotion_classifier.py
# @Software: PyCharm

from keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping
from keras.callbacks import ReduceLROnPlateau
from utils.path_IDs import PathIds
from utils.data_generator import DataGenerator

from models.Convlstm import ConvlstmV2
from keras.optimizers import Adam

BATCH_SIZE = 32
NUM_EPOCHS = 1000
VERBOSE = 1
PATIENCE = 50
INPUT_SHAPE = (64, 9, 9, 4)
NUM_CLASSES = 2

base_path = 'E:\\Programming\\python program\\EEG_emotion\\DEAP\\saved_models\\'

model = ConvlstmV2(INPUT_SHAPE, NUM_CLASSES)
adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
model.compile(optimizer='adam', loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

data_type = 'raw_data'
train_mode = ['data_2', 'data_4']
train_type = ['valence', 'arousal']

for mode in train_mode:
    print('Traing mode:', mode)

    for type in train_type:
        print('Training type:', type)

        log_file_path = base_path + mode + '\\' + type  + '\\'+ '_EEG_emotion_training.log'
        csv_logger = CSVLogger(log_file_path, append=False)
        early_stop = EarlyStopping('val_loss', patience=PATIENCE)
        reduce_lr = ReduceLROnPlateau('val_loss', factor=0.1,
                                      patience=int(PATIENCE / 5), verbose=1)
        trained_models_path = base_path + mode + '\\' + type  + '\\'+ '_ConvLstmV2'
        model_names = trained_models_path + '.{epoch:02d}-{val_acc: .2f}.hdf5'
        model_checkpoint = ModelCheckpoint(model_names, 'val_loss', verbose=1,
                                           save_best_only=True)
        callbacks = [model_checkpoint, csv_logger]

        # loading dataset
        train_ids_loader = PathIds('Train', data_type, mode)
        test_ids_loader = PathIds('Validation', data_type, mode)
        train_path_ids = train_ids_loader.generator_path_IDs()
        test_path_ids = test_ids_loader.generator_path_IDs()

        training_generator = DataGenerator(type, mode, train_path_ids, BATCH_SIZE)
        validation_generator = DataGenerator(type, mode, test_path_ids, BATCH_SIZE)

        model.fit_generator(generator = training_generator,
                            steps_per_epoch= len(train_path_ids) / BATCH_SIZE,
                            epochs=NUM_EPOCHS,
                            verbose=1, callbacks=callbacks,
                            validation_data = validation_generator,
                            validation_steps = len(test_path_ids) / BATCH_SIZE
                            )

        # _, accuracy = model.evaluate_generator(generator= validation_generator,
        #                                        steps=len(test_path_ids) / BATCH_SIZE)
        # print(accuracy)
