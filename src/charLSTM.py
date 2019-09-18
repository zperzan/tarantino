from keras.models import Sequential
from keras.layers import Dense, LSTM, Bidirectional
from keras.callbacks import LambdaCallback, ModelCheckpoint, CSVLogger, EarlyStopping
import numpy as np
import random
import sys
import os
import io

def unembed(array, chars):
    """Helper function to take an array of one-hot character vectors
    turn it into text.
        Args: 
            array (n x p): array of n one-hot character vectors, each 
                           of length p
            chars (list[str]): ordered list of the unqiue characters
                               in the training set

        Returns:
            sentence: The text that was embedded as an array
    """
    sentence = ''
    char_indices = dict((c, i) for i, c in enumerate(chars))
    indices_char = dict((i, c) for i, c in enumerate(chars))
    for vector in array:
        # Get index of True in one-hot vector
        ind = np.where(vector == True)[0][0]
        sentence += indices_char[ind]
        
    return sentence

class textGenModel():

    def __init__(self, chars, name="textGenModel", layers=1, hidden_nodes=256, 
                 bidirectional=False, optimizer='adam', dropout=0.2,
                 savepath = ""):
        """Create a new model to handle character-level text generation.

        The wrapper class will hold all states with respect to model building.

        Args:
            chars (list[str]): List of all characters used in the model.
            name (str): The name of the instance, to be used in callback file names
            layers (int): The number of LSTM layers to include in the model.
            hidden_nodes (int): The number of hidden units in each LSTM.
            bidirectional (bool): Whether or not to process the data using a
                bidirectional or single-directional LSTM.
            optimizer (str | keras.optimizers.Optimizer): The optimizer to use
                when optimizing the loss on the training set using Stochastic
                Gradient Descent (SGD).
            dropout (float): Recurrent dropout to include in each LSTM layer.
            savepath (str): Path to which the checkpoint files are written. Default 
                            is the current directory, but creates a new one if savepath
                            does not exist.
        """
        # Store characters
        self.chars = chars
        
        # Store model hyperparameters
        self.name = name
        self.layers = layers
        self.hidden_nodes = hidden_nodes
        self.bidirectional = bidirectional
        self.optimizer = optimizer
        self.dropout = dropout
        
        # Check if savepath exists
        if len(savepath) == 0:
            # current dir if not specified
            self.savepath = os.getcwd()
        else:
            # Check if this is specified from root or relative to current directory
            if savepath[0] != "/":
                savepath = os.path.join(os.getcwd(), savepath)
            
            # Create it if does not exist
            if not os.path.exists(savepath):
                print("WARNING -- Directory "+savepath+" not found.")
                print("Creating new directory")
                os.makedirs(savepath)
            
            # Store for later
            self.savepath = savepath
        
    def build_model(self):
        """Builds a new keras model for the class.

        Returns:
            keras.models.Model: A built and compiled Keras model 
        """
        # Vocab size (# of unique characters within the data)
        num_chars = len(self.chars)
        
        print('Building model with following parameters...')
        print()
        print('Layers: ', self.layers)
        print('Bidirectional: ', self.bidirectional)
        print('Hidden Nodes: ', self.hidden_nodes)
        print('Dropout: ', self.dropout)
        model = Sequential()
        for i in range(self.layers - 1):
            if self.bidirectional:
                model.add(Bidirectional(LSTM(self.hidden_nodes, 
                               input_shape=(self.seqlen, num_chars), 
                               recurrent_dropout=self.dropout,
                               return_sequences=True)))
            else:
                model.add(LSTM(self.hidden_nodes, 
                               input_shape=(self.seqlen, num_chars), 
                               recurrent_dropout=self.dropout,
                               return_sequences=True))

        if self.bidirectional:
            model.add(Bidirectional(LSTM(self.hidden_nodes, 
                                         input_shape=(self.seqlen, num_chars),
                                         recurrent_dropout=self.dropout)))
        else:
            model.add(LSTM(self.hidden_nodes, 
                           input_shape=(self.seqlen, num_chars),
                           recurrent_dropout=self.dropout))

        # Add the fully-connected layer that takes it from the hidden layer(s) to the 
        # output
        model.add(Dense(num_chars, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])
        
        return model

    def sample(self, preds, diversity=1.0):
        """Utility that samples from a probability array and 
        returns the index of the highest probability
        
        Args:
            preds (array): array of softmax probabilities
            diversity (float): amount to scale the input softmax array before
                               sampling from it. Larger values create more parity
                               between probabilities, smaller values cause the 
                               highest probability to stand out even more
        
        """
        preds = np.asarray(preds).astype('float64')
        preds = np.log(preds) / diversity # Scale probabilities
        
        # Take the softmax of the scaled probabilities
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        
        # Draw a random sample from the scaled softmax
        probas = np.random.multinomial(1, preds, 1)
        return np.argmax(probas)

    def load_params(self, path):
        """Load parameter weights from a pre-trained model.
        
        Args:
            path (str): path to the hdf5 keras file with parameter weights
            
        Returns:
            self
        """
        self.model.build((None, self.seqlen, len(self.chars)))
        self.model.load_weights(path)
        
        print("Loaded a keras model with the following parameters:")
        self.model.summary()

    def generate_text(self, seed, genlen=300, diversity=1.0):
        """Generate text from a model
        
        Args:
            seed (str): Seed from which to generate text
            genlen (int): Number of characters of generated text
            diversity (float): The log scaling of the Softmax during sampling.
                               This controls the diversity of the generated text.

        Returns:
            generated (str): Generated text of length genlen
        """
        
        num_chars = len(self.chars)
        
        # Get character embedding
        char_indices = dict((c, i) for i, c in enumerate(self.chars))
        indices_char = dict((i, c) for i, c in enumerate(self.chars))
        
        generated = ''
        
        for i in range(genlen):
            x_pred = np.zeros((1, self.seqlen, num_chars))
            for t, char in enumerate(seed):
                x_pred[0, t, char_indices[char]] = 1.

            preds = self.model.predict(x_pred, verbose=0)[0]
            next_index = self.sample(preds, diversity)
            next_char = indices_char[next_index]

            generated += next_char
            seed = seed[1:] + next_char
        
        return generated
        
    def on_epoch_end(self, epoch, _):
        # Custom callback invoked at end of each epoch. Prints generated text to file.
        filename = os.path.join(self.savepath, self.name+"_genText.txt")
        with io.open(filename, 'a') as f:
            print(file=f)
            print('Generating text after Epoch: %d' % epoch, file=f)
            
            index = random.randint(0, self.x_train.shape[0])
            sentence = unembed(self.x_train[index], self.chars)
            print('Generating text from seed: \n"' + sentence + '"', file=f)

            for diversity in [0.2, 0.5, 1.0, 1.2]:
                print('----- Temperature:', diversity, file=f)
                print(file=f)

                generated = sentence
                generated += self.generate_text(sentence, diversity=diversity)
                print(generated, file=f)
                print(file=f)

    def fit(self, x_train, y_train, batch_size=250, validation_data=None, **kwargs):
        """Trains the model 

        Args:
            x_train (array): n x m x p array of n training examples, each with m time steps and
                             p features (in our case, this is the number of characters)
            y_train (array): n x p array of n training examples with of p output vectors
            batch_size (int): number of examples to include in each batch for SGD
            validation_data (tuple): tuple of the form (x_test, y_test) for use in model validation
            **kwargs: Passed to keras.Model.fit

        Returns:
            self

        Raises:
            ValueError: If validation data is not of the correct format.
        """
        
        # Store for use in self.on_epoch_end
        self.x_train = x_train
        
        # Vocab size (# of unique characters within the data)
        num_chars = len(self.chars)
        # Length of the training sequences
        self.seqlen = x_train.shape[1]
        
        if validation_data:
            # Process validation data the same way we process our training data
            if not len(validation_data) == 2:
                raise ValueError('Validation data must be a tuple (X, y)')
            
        # Construct our Keras model
        self.model = self.build_model()

        # Set up callbacks
        # Callback #1 (print generated text to file)
        cb1 = LambdaCallback(on_epoch_end = self.on_epoch_end)
        # Callback #2 (save periodic weights)
        filepath = os.path.join(self.savepath, self.name+"_periodic_weights.{epoch:02d}-{val_acc:.2f}.hdf5")
        cb2 = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=False, mode='max', period=1)
        # Callback #3 (save best weights)
        filepath = os.path.join(self.savepath, self.name+"_best_weights.{epoch:02d}-{val_acc:.2f}.hdf5")
        cb3 = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max', period=1)
        # Callback #4 (save epoch accuracy and loss)
        filepath = os.path.join(self.savepath, self.name+"_traininglog.csv")
        cb4 = CSVLogger(filepath)
        # Callback #5 (early stopping)
        cb5 = EarlyStopping(monitor='val_acc', min_delta=0.01, patience=3, verbose=1, mode='max', restore_best_weights=True)
        
        # Train the model with callbacks and early stopping
        self.model.fit(x_train, y_train, 
                       validation_data=validation_data, 
                       callbacks = [cb1, cb2, cb3, cb4, cb5],
                       epochs=20,
                       **kwargs)
        
        return self
