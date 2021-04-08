import hr_cams
from tensorflow.keras.applications import ResNet50

X_train = 'path/to/traindata'
X_test = 'path/to/testdata'
model = ResNet50(input_tensor=ip, include_top=False, weights='imagenet', input_shape=[256, 256, 3])  # set model as your trained model
saved_weights = '../path/to/weights.hdf5'
layer_ids = [-1, -3, -5]  # Add layer_ids to calculate cams from
image = 'path/to/image'

hr_cams(model, init_weights=saved_weights, image=image, train_data=X_train, test_data=X_test, layer_ids=layer_ids)

"""
    Training the HR-Model
"""

hr_model = new_model_hr()
hr_filename = os.path.join('logs', 'hr_cam.csv')
hr_filepath = os.path.join('weights', 'hr_cam.hdf5')
hr_csv_log = CSVLogger(hr_filename, separator=',', append=True)
hr_checkpoint = ModelCheckpoint(hr_filepath, save_best_only=False)
hr_rl = ReduceLROnPlateau(monitor='val_accuracy', patience=10, min_delta=0.001, cooldown=5,)
hr_tb = TensorBoard('./logs', histogram_freq=0)
hr_callbacks_list = [hr_csv_log,
                     hr_checkpoint,
                     hr_rl,
                     hr_tb,
                     ]
hr_model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0003), loss='categorical_crossentropy', metrics=['accuracy'])
hr_model.load_weights(self.init_weights, by_name=True)
train_steps = len(self.train_datagen)
test_steps = len(self.test_datagen)

epochs = 20

HR_H = hr_model.fit(self.train_datagen, steps_per_epoch=train_steps, epochs=epochs,
                    verbose=1, validation_data=self.test_datagen, validation_steps=test_steps, callbacks=hr_callbacks_list)
hr_model.save_weights(hr_filepath)

display_hr_cams(hr_model, hr_filepath)
