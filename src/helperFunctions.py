import datetime
import pytz
import math
import glob
import pickle
import numpy as np
import matplotlib.pyplot as plt
from music21 import converter, instrument, note, chord, stream
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Activation, BatchNormalization, Embedding
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
import time
import math
import os

# CONSTANTS

# data folder location
curr = os.path.dirname(__file__)
DATA_FOLDER = os.path.join(curr, "../data/")

# ---------------- General helper functions ----------------

# function to help print time elapsed
def stringTime(start, end, show_ms=False):
    """
    Formats a given number of seconds into hours, minutes, seconds and milliseconds.

    Args:
        start (float): The start time (in seconds)
        end (float): The end time (in seconds)

    Returns:
        t (str): The time elapsed between start and end, formatted nicely.
    """
    h = "{0:.0f}".format((end-start)//3600)
    m = "{0:.0f}".format(((end-start)%3600)//60)
    s = "{0:.0f}".format(math.floor(((end-start)%3600)%60))
    ms = "{0:.2f}".format((((end-start)%3600)%60 - math.floor(((end-start)%3600)%60))*1000) # remember s = math.floor(((end-start)%3600)%60
    h_str = f"{h} hour{'' if float(h)==1 else 's'}"
    m_str = f"{'' if float(h)==0 else ', '}{m} minute{'' if float(m)==1 else 's'}"
    s_str = f"{'' if (float(h)==0 and float(m)==0) else ', '}{s} second{'' if float(s)==1 else 's'}"
    ms_str = f"{'' if (float(h)==0 and float(m)==0 and float(s)==0) else ', '}{ms} ms"

    t = f"{h_str if float(h) != 0 else ''}{m_str if float(m) != 0 else ''}{s_str if float(s) != 0 else ''}{ms_str if show_ms else ''}"
    return t

# get current time
def getTime(timezone="Canada/Eastern"):
    """
    Creates a 'now' object containing information about the current time.
    Used for logging/saving files.

    Returns:
        now (utc timezone object): Object representing the current date & time
    """
    utc_now = pytz.utc.localize(datetime.datetime.utcnow())
    now = utc_now.astimezone(pytz.timezone(timezone))
    return now


# ---------------- Project helper functions ----------------

# function to load MIDI files from for subsequent processing
def loadMidi(path=DATA_FOLDER+"training_data", verbose=False):
    midi_files = glob.glob(f'{path}/*.mid')
    start = time.time()

    all_events = []
    all_durations = []
    sizes = []
    numFlat = 0
    for file in midi_files:
        midi = converter.parse(file)
        
        # check if file has instrument parts
        try:
            parts = instrument.partitionByInstrument(midi)
            events = parts.parts[0].recurse()
        # if not, file only has notes in flat structure
        except:
            events = midi.flat.notes
            numFlat += 1

        sizes.append(len(events))

        # append to all_events
        for event in events:
            all_durations.append(event.duration.quarterLength)
            if (isinstance(event, note.Note)):
                # event is a single note, so we only want its pitch
                all_events.append(str(event.pitch))
            elif (isinstance(event, chord.Chord)):
                # event is a chord, so save it's pitches as normal order integers
                all_events.append('.'.join(str(t) for t in event.normalOrder))

    end = time.time()

    if (verbose):
        print('-----------------------------------------')
        print(f'Time elapsed: {stringTime(start, end, show_ms=True)}')
        print('')

        print(f'Number of flat structure files:  {numFlat}')
        print(f'Number of instrument part files: {len(midi_files)-numFlat}')
        print('')

    return (all_events, all_durations)

# function to create features from raw MIDI data
def createFeatures(raw_data, input_length=100):
    # get list of unique elements in raw MIDI data array
    names = sorted(set(e for e in raw_data))

    # get number of classes (i.e. number of unique elements)
    n_classes = len(names)

    # create dictionary to map elements to integers
    int_map = dict((n, i) for i, n in enumerate(names))

    # create input features and output labels
    X_train = []
    y_train = []
    for i in range(0, len(raw_data)-input_length, 1):
        melody_in = raw_data[i:i+input_length]
        note_out = raw_data[i+input_length]
        X_train.append([int_map[n] for n in melody_in])
        y_train.append(int_map[note_out])
    
    # reshape and normalize input
    X_train = np.asarray(X_train)
    X_train = X_train.reshape((-1, input_length, 1))
    X_train = X_train/float(n_classes)

    # format labels as categorical data
    y_train = np_utils.to_categorical(y_train)

    return (X_train, y_train, n_classes)

# function to create the RNN model
def createModel(n_classes, X_train):
    model = Sequential()
    model.add(LSTM(
        512,
        input_shape=(X_train.shape[1], X_train.shape[2]),
        recurrent_dropout=0.3,
        return_sequences=True
    ))
    model.add(LSTM(512, return_sequences=True, recurrent_dropout=0.3,))
    model.add(LSTM(512))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(n_classes))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    
    return model

# function to train Gen-pitch
def trainPitchRNN(model, X_train, y_train, epochs=200, batch_size=128, verbose=False):
    now = getTime()
    filename = now.strftime(f"%Y%m%d-%H%M_weights_pitch_{epochs:02d}ep.hdf5")
    checkpoint = ModelCheckpoint(
        os.path.join(DATA_FOLDER+"weights/pitch", filename),
        monitor='loss',
        verbose=verbose,
        save_best_only=True,
        save_weights_only=True,
        mode='min'
    )

    start = time.time()
    hist = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, callbacks=[checkpoint], verbose=verbose)
    end = time.time()

    print('-----------------------------------------')
    print(f'Time elapsed: {stringTime(start, end, show_ms=True)}')
    print('')

    return hist

# function to train Gen-duration
def trainDurationRNN(model, X_train, y_train, epochs=10, batch_size=128, verbose=False):
    now = getTime()
    filename = now.strftime(f"%Y%m%d-%H%M_weights_duration_{epochs:02d}ep.hdf5")
    checkpoint = ModelCheckpoint(
        os.path.join(DATA_FOLDER+"weights/duration", filename),
        monitor='loss',
        verbose=verbose,
        save_best_only=True,
        save_weights_only=True,
        mode='min'
    )

    start = time.time()
    hist = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, callbacks=[checkpoint], verbose=verbose)
    end = time.time()

    print('-----------------------------------------')
    print(f'Time elapsed: {stringTime(start, end, show_ms=True)}')
    print('')

    return hist

# function to generate a new melody given a trained model
# will generate a novel pitch sequence or a novel duration sequence
def generateNewData(model, raw_data, X_train, length=500):
    names = sorted(set(e for e in raw_data))
    n_classes = len(names)
    int_map_reverse = dict((i, n) for i, n in enumerate(names))


    data = []
    input = X_train[np.random.randint(0, len(X_train)-1)]
    start = time.time()
    for _ in range(length):
        # use random input sequence to get an output note
        X = input.reshape((1, len(input), 1))
        X = X/float(n_classes)
        pred = model.predict(X, verbose=0)
        bestIndex = np.argmax(pred)

        # append output note to melody being built
        data.append(int_map_reverse[bestIndex])

        # add note index to input and shift input forwards by one for next round
        input = np.append(input, bestIndex)
        input = input[1:len(input)]
    end = time.time()
    print('-----------------------------------------')
    print(f'Time elapsed: {stringTime(start, end, show_ms=True)}')
    print('')

    return data

# function to create a new MIDI file given a generated pitch sequence (Gen-pitch)
def createMidiPitch(pitches, midi_path=DATA_FOLDER+"output_MIDI"):
    offset = 0
    timestep = 0.5
    output_notes = []

    # create note and chord objects based on the values generated by the model
    for event in pitches:
        if ('.' in event) or event.isdigit():
            # event is a chord
            chordNotes = event.split('.')
            notes = []
            for n in chordNotes:
                new_note = note.Note(int(n))
                new_note.storedInstrument = instrument.Piano()
                notes.append(new_note)
            new_chord = chord.Chord(notes)
            new_chord.offset = offset
            output_notes.append(new_chord)
        else:
            # event is a note
            new_note = note.Note(event)
            new_note.offset = offset
            new_note.storedInstrument = instrument.Piano()
            output_notes.append(new_note)
        # increase offset each iteration so that notes do not stack
        offset += timestep

    midi_stream = stream.Stream(output_notes)
    now = getTime()
    filename = now.strftime("%Y%m%d-%H%M_genpitch.mid")
    midi_stream.write('midi', fp=os.path.join(midi_path, filename))

# function to create a new MIDI file given a generated pitch AND duration sequence (Gen-duration)
def createMidiDuration(pitches, durations, midi_path=DATA_FOLDER+"output_MIDI"):
    offset = 0
    output_notes = []

    # create note and chord objects based on the values generated by the model
    for i, event in enumerate(pitches):
        if ('.' in event) or event.isdigit():
            # event is a chord
            chordNotes = event.split('.')
            notes = []
            for n in chordNotes:
                new_note = note.Note(int(n))
                new_note.storedInstrument = instrument.Piano()
                new_chord.duration.quarterLength = durations[i]
                notes.append(new_note)
            new_chord = chord.Chord(notes)
            new_chord.offset = offset
            output_notes.append(new_chord)
        else:
            # event is a note
            new_note = note.Note(event)
            new_note.offset = offset
            new_note.storedInstrument = instrument.Piano()
            new_note.duration.quarterLength = durations[i]
            output_notes.append(new_note)
        # increase offset each iteration so that notes do not stack
        offset += durations[i]

    midi_stream = stream.Stream(output_notes)
    now = getTime()
    filename = now.strftime("%Y%m%d-%H%M_genduration.mid")
    midi_stream.write('midi', fp=os.path.join(midi_path, filename))