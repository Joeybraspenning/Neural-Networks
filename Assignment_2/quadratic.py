from __future__ import print_function
from keras.models import Sequential
from keras import layers
from keras import callbacks
from keras.callbacks import CSVLogger
from keras.layers.normalization import BatchNormalization
import numpy as np
from six.moves import range

################################################################################
################################################################################
################################################################################

class CharacterTable(object):
    """Given a set of characters:
    + Encode them to a one-hot integer representation
    + Decode the one-hot or integer representation to their character output
    + Decode a vector of probabilities to their character output
    """
    def __init__(self, chars):
        """Initialize character table.

        # Arguments
            chars: Characters that can appear in the input.
        """
        self.chars = sorted(set(chars))
        self.char_indices = dict((c, i) for i, c in enumerate(self.chars))
        self.indices_char = dict((i, c) for i, c in enumerate(self.chars))

    def encode(self, C, num_rows):
        """One-hot encode given string C.

        # Arguments
            C: string, to be encoded.
            num_rows: Number of rows in the returned one-hot encoding. This is
                used to keep the # of rows for each data the same.
        """
        x = np.zeros((num_rows, len(self.chars)))
        for i, c in enumerate(C):
            x[i, self.char_indices[c]] = 1
        return x

    def decode(self, x, calc_argmax=True):
        """Decode the given vector or 2D array to their character output.

        # Arguments
            x: A vector or a 2D array of probabilities or one-hot representations;
                or a vector of character indices (used with `calc_argmax=False`).
            calc_argmax: Whether to find the character index with maximum
                probability, defaults to `True`.
        """
        if calc_argmax:
            x = x.argmax(axis=-1)
        return ''.join(self.indices_char[x] for x in x)

################################################################################
################################################################################
################################################################################

def generate_data_set(n_items, input_len, n_decimals):

    while len(questions) < n_items:

        # Sample numbers uniformly in log space
        exp = np.random.uniform(1,input_len-0.1)
        number = int(10**exp)

        # Skip the number if it has been encounterd before
        if number in seen:
            continue
        seen.add(number)

        # Build the input and output strings
        ans = '{}'.format(number**2)
        ans += ' ' * (input_len + input_len - len(ans))
        query = str(number)

        # Make sure that all query entries have the same length
        query += ' ' * (input_len - len(query))

        questions.append(query)
        expected.append(ans)

    print('Total number of questions:', len(questions))

    return questions, expected

################################################################################
################################################################################
################################################################################

# Tunable parameters
TRAINING_SIZE = 5000
TEST_SIZE = 1000
INPUT_LEN = 5 # The maximum number of digits in the input integers
DECIMALS = 3 # the number of decimals in the scientific notation

# This number is fixed
OUTPUT_LEN = INPUT_LEN + INPUT_LEN

chars = '0123456789 '
ctable = CharacterTable(chars)

questions = []
expected = []
seen = set()
print('Generating data...')

questions, expected = generate_data_set(TRAINING_SIZE+TEST_SIZE, INPUT_LEN, DECIMALS)

################################################################################
################################################################################
################################################################################

print('Vectorization...')
x = np.zeros((len(questions), INPUT_LEN, len(chars)), dtype=np.bool)
y = np.zeros((len(questions), OUTPUT_LEN, len(chars)), dtype=np.bool)

# Encode all inputs and outputs (i.e. turn strings into matrices)
for i, sentence in enumerate(questions):
    x[i] = ctable.encode(sentence, INPUT_LEN)
for i, sentence in enumerate(expected):
    y[i] = ctable.encode(sentence, OUTPUT_LEN)

# Shuffle (x, y) in unison as the later parts of x will almost all be larger
# digits.
indices = np.arange(len(y))
np.random.shuffle(indices)
x = x[indices]
y = y[indices]

# Split the data over the training set (90% training and 10% validation) and the test set
split1 = int(0.9*TRAINING_SIZE)
split2 = TRAINING_SIZE

(x_train, x_val, x_test) = x[:split1], x[split1:split2], x[split2:]
(y_train, y_val, y_test) = y[:split1], y[split1:split2], y[split2:]

print('Training Data:')
print(x_train.shape)
print(y_train.shape)

print('Validation Data:')
print(x_val.shape)
print(y_val.shape)

print('Test Data:')
print(x_val.shape)
print(y_val.shape)

################################################################################
################################################################################
################################################################################

# Try replacing GRU, or SimpleRNN.
RNN = layers.LSTM
HIDDEN_SIZE = 128
BATCH_SIZE = 128
LAYERS = 1

print('Build model...')
model = Sequential()
# "Encode" the input sequence using an RNN, producing an output of HIDDEN_SIZE.
# Note: In a situation where your input sequences have a variable length,
# use input_shape=(None, num_feature).
model.add(RNN(HIDDEN_SIZE, input_shape=(INPUT_LEN, len(chars)), activation=layers.PReLU(), recurrent_activation='sigmoid',\
                dropout=0.1, recurrent_dropout=0.05))
model.add(BatchNormalization(center=True, scale=True))
# As the decoder RNN's input, repeatedly provide with the last output of
# RNN for each time step. Repeat 'DIGITS + 1' times as that's the maximum
# length of output, e.g., when DIGITS=3, max output is 999+999=1998.
model.add(layers.RepeatVector(OUTPUT_LEN))
model.add(BatchNormalization(center=True, scale=True))
# The decoder RNN could be multiple layers stacked or a single layer.
for _ in range(LAYERS):
    # By setting return_sequences to True, return not only the last output but
    # all the outputs so far in the form of (num_samples, timesteps,
    # output_dim). This is necessary as TimeDistributed in the below expects
    # the first dimension to be the timesteps.
    model.add(RNN(HIDDEN_SIZE, return_sequences=True, activation=layers.PReLU(), recurrent_activation='sigmoid',\
                    dropout=0.1, recurrent_dropout = 0.05))
    model.add(BatchNormalization(center=True, scale=True))
# Apply a dense layer to the every temporal slice of an input. For each of step
# of the output sequence, decide which character should be chosen.
model.add(layers.TimeDistributed(layers.Dense(len(chars), activation='softmax')))
model.compile(loss='categorical_crossentropy',
              optimizer='Nadam',
              metrics=['accuracy'])
model.summary()

training_accuracies = []
training_losses = []

# Train the model each generation and show predictions against the validation
# dataset.
for iteration in range(1, 200):
    print()
    print('-' * 50)
    print('Iteration', iteration)
    hist = model.fit(x_train, y_train,
              batch_size=BATCH_SIZE,
              epochs=1,
              validation_data=(x_val, y_val))
    # Select 10 samples from the validation set at random so we can visualize
    # errors.

    #print(hist.history)
    training_accuracies.append([hist.history['acc'][0], hist.history['val_acc'][0]])
    training_losses.append([hist.history['loss'][0], hist.history['val_loss'][0]])

    for i in range(10):
        ind = np.random.randint(0, len(x_val))
        rowx, rowy = x_val[np.array([ind])], y_val[np.array([ind])]
        preds = model.predict_classes(rowx, verbose=0)
        q = ctable.decode(rowx[0])
        correct = ctable.decode(rowy[0])
        guess = ctable.decode(preds[0], calc_argmax=False)

        print('Q', q, end=' ')
        print('T', correct, end=' ')
        if correct == guess:
            print('OK', end=' ')
        else:
            print('..', end=' ')
        print(guess)

################################################################################
################################################################################
################################################################################

scores = model.evaluate(x_test, y_test, verbose=1)
print('-----------------------------------------')
print('Test accuracy: ', scores[1])

np.save('training_accuracies', np.array(training_accuracies))
np.save('training_losses', np.array(training_losses))
