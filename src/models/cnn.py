import tensorflow as tf

class CNNModel(tf.keras.Model):
    """
    Convolutional Neural Network (CNN) model implemented using TensorFlow.

    Parameters:
        num_classes (int): The number of classes in the classification problem.

    Methods:
        call(inputs):
            Defines the forward pass of the CNN model.

    Example usage:
        # Assuming you have your training data `X_train` and corresponding labels `y_train`
        num_classes = 10  # Number of classes in your classification problem
        model = CNNModel(num_classes)

        # Compile the model
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        # Train the model
        model.fit(X_train, y_train, epochs=10, batch_size=32)

        # Make predictions
        predictions = model.predict(X_test)
    """

    def __init__(self, num_classes):
        super(CNNModel, self).__init__()

        # Define the layers for the CNN model
        self.reshape = tf.keras.layers.Reshape((26, 1), input_shape=(26,))
        self.conv1 = tf.keras.layers.Conv1D(128, kernel_size=3, activation='relu')
        self.maxpool1 = tf.keras.layers.MaxPooling1D(pool_size=2)
        self.conv2 = tf.keras.layers.Conv1D(64, kernel_size=3, activation='relu')
        self.maxpool2 = tf.keras.layers.MaxPooling1D(pool_size=2)
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(128, activation='relu')
        self.dense2 = tf.keras.layers.Dense(num_classes, activation='softmax')

    def call(self, inputs):
        """
        Defines the forward pass of the CNN model.

        Parameters:
            inputs (Tensor): The input tensor to the model.

        Returns:
            Tensor: The output tensor of the model.
        """
        x = self.reshape(inputs)
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.flatten(x)
        x = self.dense1(x)
        output = self.dense2(x)
        return output
