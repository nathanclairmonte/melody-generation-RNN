import helperFunctions as hf

if (__name__=="__main__"):
    # load data
    (all_events, _) = hf.loadMidi()
    
    # create features
    (X_train, y_train, n_classes) = hf.createFeatures(all_events)

    # create model
    model = hf.createModel(n_classes, X_train)

    # train model
    hf.trainPitchRNN(model, X_train, y_train)