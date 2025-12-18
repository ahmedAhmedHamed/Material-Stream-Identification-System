احمد احمد حامد احمد #
# 20211003

# foreword
I've decided to leave in all the parts of the code that are left unused for the sake of showing effort.
the parts that the user is expected to interact with are:
- for mobile server
- - api_server.py
- - android_jetpack_project

- for feature extraction
- - standalone_scripts.run_feature_extractor_v2_dataset.py

- for data cleanup
- - standalone_scripts.clean_corrupted_data.py

- for testing:
- - test.py

all other files are either internal files or are completely unused.

# performance reports:

in manual vector extraction: 
adding large features is a bad idea; adding HOB to the features always tanked the accuracy by a large margin.
adding many small features did not change the accuracy by a huge margin.
any of the currently used features is enough for the model to have 60% accuracy
all of them together give around 85% accuracy.

if I were to spend more time on this I would likely add more vectors to the extraction.

in data augmentation:

I tried to make it so that I increase all the classes to be 2X the biggest class but that did not produce adequate results.
Then tried to make it so that I undersample only paper and oversample the rest, that produced better results.
currently it is made so that the target is the median of the dataset and the data augmentation increases the size and all of them are over or under sampled to reach that goal.

some data augmentations reduced accuracy while some did not have much of an effect.
removed the ones that reduced accuracy, kept the rest.

adding a lot of data did not increase the model accuracy much, but did increase the time it took to infer and train, so I left it on 1.0x at the end.


in SVM and KNN, instead of manually experimented I made it so that the script programmatically tries most of the sensible combinations.
in KNN it always ends up being that N = 1 is the correct one
in SVM it always ends up that the correct parameters is found in iterations 20 through 25

# components

- standalone scripts
- - clean_corrupted_data.py: deletes the 0 bytes files that existed in the dataset originally
- - run_feature_extractor_v2_dataset.py: runs the newest and final feature extractor. (configurable at top of file)
- - - the feature extractor creates features_train.npz and features_val.npz, both created at the project root
- - - it can be configured for how much data to add and which augmentations to apply.

- classifiers
- - KNN
- - - knn uses the pre-existing features that were created and trains a knn model for it
- - - it loops over values starting at 1 ending at 5 and adding 5 each time and saves the best one
- - - also saves knn_accuracies json for more investigation on why the best model is the best.
- - - KNN is saved in KNN_best
- - SVM
- - - uses the pre-existing features that were created and trains an SVM model for it.
- - - loops over some values that were created.
- - - they are three arrays for each of the parameters and it brute forces them and saves the best one.
- - - SVM is saved in SVM_{number}

- vector extraction
- - in this directory there are three iterations of vector extraction.
- - the first one implemented was feature_extractor.py; it extracts the features using some functions and concatenates them and returns them.
- - the second one implemented was deep_feature_extractor.py; it is by far the best performing one but was not used in the final model as it was too resource hungry.
- - the third and final one was feature_extractor_v2*.py this is almost the same as the first one but more configurable.
- - - experimented with adding and removing features and augmentations.

- pyproject.toml, uv.lock, .python-version:
- - this project uses uv as its version manager.
- - install uv then run uv sync and everything will be ready. (except for the mobile application.)

- api_server.py
- - simple server that takes the required model to run and the image to run it with.
- - - saves the frames in /frames for debugging purposes.

- android_jetpack_project
- - houses the mobile application.
- - mobile application requires the server to be running.
- - it is a gradle jetpack compose app.

- test.py
- - as requested.

