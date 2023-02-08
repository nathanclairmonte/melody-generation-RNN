import os
import helperFunctions as hf

# CONSTANTS

# data folder location
curr = os.path.dirname(__file__)
DATA_FOLDER = os.path.join(curr, "../data/")

if (__name__=="__main__"):
    # load data
    (all_events, _) = hf.loadMidi()
    
    # create features
    (X_train, y_train, n_classes) = hf.createFeatures(all_events)

    # create model
    model = hf.createModel(n_classes, X_train)

    # load weights
    weights_filename = "20201124_pitch_weights.hdf5"
    model.load_weights(DATA_FOLDER + f"weights/pitch/{weights_filename}")

    # save MIDI file
    pitches = hf.generateNewData(model, all_events, X_train)
    hf.createMidiPitch(pitches)