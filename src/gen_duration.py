import os
import helperFunctions as hf

# CONSTANTS

# data folder location
curr = os.path.dirname(__file__)
DATA_FOLDER = os.path.join(curr, "../data/")

if (__name__=="__main__"):
    # load data
    (all_events, all_durations) = hf.loadMidi()
    
    # create features
    (X_train_pitch, y_train_pitch, n_classes_pitch) = hf.createFeatures(all_events)
    (X_train_dur, y_train_dur, n_classes_dur) = hf.createFeatures(all_durations)

    # create models
    model_pitch = hf.createModel(n_classes_pitch, X_train_pitch)
    model_dur = hf.createModel(n_classes_dur, X_train_dur)

    # load weights
    pitch_weights_filename = "20201124_pitch_weights.hdf5"
    duration_weights_filename = "20201124_duration_weights.hdf5"
    model_pitch.load_weights(DATA_FOLDER + f"weights/pitch/{pitch_weights_filename}")
    model_dur.load_weights(DATA_FOLDER + f"weights/duration/{duration_weights_filename}")

    # save MIDI file
    pitches = hf.generateNewData(model_pitch, all_events, X_train_pitch)
    durations = hf.generateNewData(model_dur, all_durations, X_train_dur)
    hf.createMidiDuration(pitches, durations)