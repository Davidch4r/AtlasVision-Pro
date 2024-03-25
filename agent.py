import tensorflow as tf
import numpy as np
import os 

class Agent:
    def __init__(self, shape, extreme_points, path=None, backup_path=None):
        self.shape = (shape[0], shape[1], 1) # input shape
        self.path = path
        self.extreme_points = extreme_points # {'N': 50, 'S': 24, 'E': -66, 'W': -126}
        if path:
            self.load_model()
        else:
            self.model = self.build_model()
            self.path = "model.keras"
            self.model.save(self.path)

        # Create a backup of the model
        if backup_path:
            self.model.save(backup_path)

    def build_model(self):
        model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=self.shape),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(2, activation='sigmoid')
        ])
        model.build((None,) + self.shape)
        return model
    
    # TODO: Generalize this function to deep_learn_batch
    def deep_learn(self, image, correct_coords):
        input_image = image.get_image()
        input_image = input_image.reshape((1,) + input_image.shape) 
        
        learning_rate = 0.0001
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        
        self.model.compile(optimizer=optimizer, loss='mean_squared_error')
        self.model.fit(input_image, correct_coords.reshape((1, 2)), epochs=1)

    # def predict(self, image):
    #     input_image = image.get_image()
    #     input_image = input_image.reshape((1,) + input_image.shape)
    #     prediction = self.model.predict(input_image)
    #     return prediction 
        
    def get_grad_cam_heatmap(self, img_array, layer_name='conv2d_2'):
        # Get the last conv layer from the model
        last_conv_layer = self.model.get_layer(layer_name)
        last_conv_layer_model = tf.keras.models.Model(self.model.inputs, last_conv_layer.output)
        
        # Create a model that maps the last conv layer's output to the model's output
        classifier_input = tf.keras.Input(shape=last_conv_layer.output.shape[1:])
        x = classifier_input
        for layer in self.model.layers[self.model.layers.index(last_conv_layer) + 1:]:
            x = layer(x)
        classifier_model = tf.keras.models.Model(classifier_input, x)
        
        # Compute the gradient of the top predicted class for the input image
        # with respect to the activations of the last conv layer
        with tf.GradientTape() as tape:
            # Compute activations of the last conv layer and make the tape watch it
            last_conv_layer_output = last_conv_layer_model(img_array)
            tape.watch(last_conv_layer_output)
            # Compute the predictions
            preds = classifier_model(last_conv_layer_output)
            top_pred_index = tf.argmax(preds[0])
            top_class_channel = preds[:, top_pred_index]
        
        grads = tape.gradient(top_class_channel, last_conv_layer_output)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        last_conv_layer_output = last_conv_layer_output[0]
        heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)

        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        return heatmap.numpy()

    
    def predict(self, image, get_heat_map=False, layer_name='conv2d_2'):
        input_image = image.get_image()
        input_image = input_image.reshape((1,) + input_image.shape)
        prediction = self.model.predict(input_image)

        if get_heat_map:
            heatmap = self.get_grad_cam_heatmap(input_image, layer_name=layer_name)
            return prediction, heatmap
        else:
            return prediction, None
        
    def save_model(self):
        self.model.save(self.path)
        print(f"Model saved at {self.path}")

    def load_model(self):
        try:
            self.model = tf.keras.models.load_model(self.path)
        except Exception as e:
            print(f"Model not found at {self.path}")
            self.model = self.build_model()
            self.model.save(self.path)
        print(f"Model loaded from {self.path}")

    def restart(self):
        os.remove(self.path)
        print(f"Model deleted at {self.path}")
        self.model = self.build_model()
        self.model.save(self.path)



    

