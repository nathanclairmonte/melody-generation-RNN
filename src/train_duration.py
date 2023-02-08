import helperFunctions as hf

if (__name__=="__main__"):
    # load data
    (all_events, all_durations) = hf.loadMidi()
    
    # create features
    (X_train_pitch, y_train_pitch, n_classes_pitch) = hf.createFeatures(all_events)
    (X_train_dur, y_train_dur, n_classes_dur) = hf.createFeatures(all_durations)

    # create models
    model_pitch = hf.createModel(n_classes_pitch, X_train_pitch)
    model_dur = hf.createModel(n_classes_dur, X_train_dur)

    # train models
    hf.trainPitchRNN(model_pitch, X_train_pitch, y_train_pitch)
    hf.trainDurationRNN(model_dur, X_train_dur, y_train_dur)