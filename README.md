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

