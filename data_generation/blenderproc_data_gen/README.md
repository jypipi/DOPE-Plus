# Synthetic Data Generation with Blenderproc

## Installation
Blenderproc can be installed with pip:
```
pip install blenderproc
```
If you run into troubles, please consult the [project's own github page](https://github.com/DLR-RM/BlenderProc).


## Usage

[Blenderproc](https://github.com/DLR-RM/BlenderProc) is intended to create a single scene and render multiple frames of it. Adding and removing objects (such as varying the number of distractors) will cause memory bloat and poor performance.  To avoid this issue, we use a batching script (`run_blenderproc_datagen.py`) to run a standalone blenderproc script several times.


### Usage example:

Sampled command to run the blenderproc script:
```
./run_blenderproc_datagen.py --path_single_obj models/Cookies/textured.obj --distractors_folder google_scanned_models/ --backgrounds_folder ../dome_hdri_haven/ --width 512 --height 512 --object_class Cookies --outf cookies_imgs/  --nb_runs 8 --nb_frames 250 --nb_distractors 3 --nb_objects 1 --scale 1 --start_folder 1
```

Parameters of the top-level script can be shown by running
```
python ./run_blenderproc_datagen.py --help
```

Note that, as a blenderproc script, `generate_training_data.py` cannot be invoked with Python. It must be run via the `blenderproc` launch script.  To discover its command-line parameters, you must look
at the source-code itself; `blenderproc run ./generate_training_data.py --help` will not report
them properly.

Blenderproc searches for python modules in a different order than when invoking Python by itself.  If you run into an issue where `generate_training_data.py` fails to import modules that you have installed, you may have to re-install them via blenderproc; e.g. `blenderproc pip install pyquaternion`.

