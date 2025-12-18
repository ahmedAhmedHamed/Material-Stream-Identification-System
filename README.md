# Material Stream Identification System

**Author:** احمد احمد حامد احمد  
**ID:** 20211003

---

## Foreword

I've decided to leave in all the parts of the code that are left unused for the sake of showing effort.

The parts that the user is expected to interact with are:

### For Mobile Server
- `api_server.py`
- `android_jetpack_project/`

### For Feature Extraction
- `standalone_scripts/run_feature_extractor_v2_dataset.py`

### For Data Cleanup
- `standalone_scripts/clean_corrupted_data.py`

### For Testing
- `test.py`

All other files are either internal files or are completely unused.

---

## Performance Reports

### Manual Vector Extraction

- Adding large features is a bad idea; adding HOB to the features always tanked the accuracy by a large margin.
- Adding many small features did not change the accuracy by a huge margin.
- Any of the currently used features is enough for the model to have **60% accuracy**.
- All of them together give around **85% accuracy**.

If I were to spend more time on this, I would likely add more vectors to the extraction.

### Data Augmentation

- I tried to make it so that I increase all the classes to be 2X the biggest class, but that did not produce adequate results.
- Then tried to make it so that I undersample only paper and oversample the rest, which produced better results.
- Currently, it is made so that the target is the median of the dataset and the data augmentation increases the size, and all of them are over or under sampled to reach that goal.

- Some data augmentations reduced accuracy while some did not have much of an effect.
- Removed the ones that reduced accuracy, kept the rest.

- Adding a lot of data did not increase the model accuracy much, but did increase the time it took to infer and train, so I left it on **1.0x** at the end.

### SVM and KNN

- Instead of manually experimenting, I made it so that the script programmatically tries most of the sensible combinations.
- In **KNN**, it always ends up being that **N = 1** is the correct one.
- In **SVM**, it always ends up that the correct parameters are found in **iterations 20 through 25**.

---

## Components

### Standalone Scripts

#### `clean_corrupted_data.py`
Deletes the 0-byte files that existed in the dataset originally.

#### `run_feature_extractor_v2_dataset.py`
Runs the newest and final feature extractor (configurable at top of file).

- The feature extractor creates `features_train.npz` and `features_val.npz`, both created at the project root.
- It can be configured for how much data to add and which augmentations to apply.

### Classifiers

#### KNN
- Uses the pre-existing features that were created and trains a KNN model for it.
- Loops over values starting at 1, ending at 5, and adding 5 each time, and saves the best one.
- Also saves `knn_accuracies.json` for more investigation on why the best model is the best.
- KNN is saved in `KNN_best/`

#### SVM
- Uses the pre-existing features that were created and trains an SVM model for it.
- Loops over some values that were created.
- They are three arrays for each of the parameters and it brute forces them and saves the best one.
- SVM is saved in `SVM_{number}/`

### Vector Extraction

In this directory, there are three iterations of vector extraction:

1. **`feature_extractor.py`** (First implementation)
   - Extracts the features using some functions and concatenates them and returns them.

2. **`deep_feature_extractor.py`** (Second implementation)
   - By far the best performing one but was not used in the final model as it was too resource hungry.

3. **`feature_extractor_v2*.py`** (Third and final implementation)
   - Almost the same as the first one but more configurable.
   - Experimented with adding and removing features and augmentations.

### Project Configuration

**`pyproject.toml`, `uv.lock`, `.python-version`**

- This project uses `uv` as its version manager.
- Install `uv`, then run `uv sync` and everything will be ready (except for the mobile application).

### API Server

**`api_server.py`**

- Simple server that takes the required model to run and the image to run it with.
- Saves the frames in `/frames` for debugging purposes.

### Mobile Application

**`android_jetpack_project/`**

- Houses the mobile application.
- Mobile application requires the server to be running.
- It is a Gradle Jetpack Compose app.

### Testing

**`test.py`**

- As requested.
