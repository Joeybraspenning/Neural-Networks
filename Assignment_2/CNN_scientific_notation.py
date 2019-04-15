from __future__ import print_function
from keras import callbacks
from keras.callbacks import CSVLogger
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Reshape, Conv2D
from keras.layers import MaxPooling2D, Flatten
from keras.layers.normalization import BatchNormalization
from keras.utils import to_categorical
import tensorflow as tf
import numpy as np
from six.moves import range

################################################################################
################################################################################
################################################################################
def match(s1, s2):
    ok = False

    for c1, c2 in zip(s1, s2):
        if c1 != c2:
            if ok:
                return False
            else:
                ok = True

    return ok

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
        ans = ('{:.'+str(n_decimals)+'e}').format(number)
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
INPUT_LEN = 9 # The maximum number of digits in the input integers
DECIMALS = 3 # the number of decimals in the scientific notation

# This number is fixed
OUTPUT_LEN = 6 + DECIMALS

chars = '0123456789e+. '
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
# RNN = layers.LSTM
# HIDDEN_SIZE = 128
BATCH_SIZE = 128
# LAYERS = 1

x_train = np.expand_dims(x_train, axis=3)
y_train = np.reshape(y_train, (y_train.shape[0], y_train.shape[1]*y_train.shape[2]))
x_val = np.expand_dims(x_val, axis=3)
y_val = np.reshape(y_val, (y_val.shape[0], y_val.shape[1]*y_val.shape[2]))


# x_train = to_categorical(x_train)
# y_train = to_categorical(y_train)
# x_val = to_categorical(x_val)
# y_val = to_categorical(y_val)

print('Training Data:')
print(x_train.shape)
print(y_train.shape)

print('Validation Data:')
print(x_val.shape)
print(y_val.shape)

print('Test Data:')
print(x_val.shape)
print(y_val.shape)



print('Build model...')
model = Sequential()
# "Encode" the input sequence using an RNN, producing an output of HIDDEN_SIZE.
# Note: In a situation where your input sequences have a variable length,
# use input_shape=(None, num_feature).
model.add(Conv2D(64, input_shape = (INPUT_LEN, len(chars), 1), kernel_size=(5, 5),\
             padding='same'))
model.add(BatchNormalization(center=True, scale=True))
model.add(Activation('sigmoid'))
model.add(Dropout(0.25))


model.add(Conv2D(64, (3, 3), padding='same'))
model.add(BatchNormalization(center=True, scale=True))
model.add(Activation('sigmoid'))
model.add(Dropout(0.25))

model.add(Conv2D(32, (3, 3), padding='same'))
model.add(BatchNormalization(center=True, scale=True))
model.add(Activation('sigmoid'))
model.add(Dropout(0.25))

model.add(Conv2D(1, (5, 5), padding='same'))
model.add(BatchNormalization(center=True, scale=True))
model.add(Activation('softmax'))

model.add(Flatten())

model.compile(loss='categorical_crossentropy',
              optimizer='Nadam',
              metrics=['accuracy'])
model.summary()


# hist = model.fit(x_train, y_train,
#           batch_size=BATCH_SIZE,
#           epochs=100,
#           validation_data=(x_val, y_val))

training_accuracies = []
training_losses = []
training_precisions = []
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
        rowx = np.squeeze(rowx, axis=3)
        rowy = np.squeeze(rowy, axis=3)
        preds = np.squeeze(preds, axis=3)
        #print(type(preds), type(np.array(preds)), preds.shape, preds[0].shape)
        q = ctable.decode(rowx[0])
        correct = ctable.decode(rowy[0])
        #print(rowy[0], preds[0])
        guess = ctable.decode(preds[0])#, calc_argmax=False)

        print('Q', q, end=' ')
        print('T', correct, end=' ')
        if correct == guess:
            print('OK', end=' ')
        else:
            print('..', end=' ')
        print(guess)
     
'''   
    full, one_off = 0, 0
    predict = model.predict_classes(x_val, verbose=0)
    for i in range(len(x_val)):
        correct = ctable.decode(y_val[i])
        guess = ctable.decode(predict[i])#, calc_argmax=False)
        if correct == guess:
            full += 1
        elif match(correct, guess):
            one_off += 1
    print('{}% of validation examples are completely correct'.format(100.
    *float(full)/len(x_val)))
    print('{}% of validation examples are one off'.format(100.*float(one_off)/len(x_val)))
        

    full_train, one_off_train = 0, 0
    predict = model.predict_classes(x_train, verbose=0)
    for i in range(len(x_train)):
        correct = ctable.decode(y_train[i])
        guess = ctable.decode(predict[i])#, calc_argmax=False)
        if correct == guess:
            full_train += 1
        elif match(correct, guess):
            one_off_train += 1
    training_precisions.append([float(full_train)/len(x_train), float(one_off_train)/len(x_train), float(full)/len(x_val), float(one_off)/len(x_val)])

################################################################################
################################################################################
################################################################################

scores = model.evaluate(x_test, y_test, verbose=1)
print('-----------------------------------------')
print('Test accuracy: ', scores[1])

# np.save('scientific_notation_accuracies_{}_{}_{}'.format(HIDDEN_SIZE, BATCH_SIZE, LAYERS), np.array(training_accuracies))
# np.save('scientific_notation_losses_{}_{}_{}'.format(HIDDEN_SIZE, BATCH_SIZE, LAYERS), np.array(training_losses))
# np.save('scientific_notation_precisions_{}_{}_{}'.format(HIDDEN_SIZE, BATCH_SIZE, LAYERS), np.array(training_precisions))
'''