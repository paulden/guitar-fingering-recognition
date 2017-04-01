# Recognition of guitar fingering using OpenCV

This repository hosts an ongoing student project in computer vision that aims at recognizing guitar finger style using OpenCV library.

## Language and modules

I used Python 3.5.2 and OpenCV 3.2.0, you'll need them installed to use these scripts.

## How to run

The project was mainly carried out using pictures in `pictures/` folder.
These images come from various sources on the Internet (YouTube videos, tutorials, etc.) and were not designed specifically to be used in computer vision (I wished) but were a good basis to begin working on.

More recently I shot various photos with a friend. These can be found in `pictures2/` folder.
These images were not specifically designed to be used in computer vision either but provide a different context.
As of April, 2nd 2017, bugs appear when using `pictures2/` images, please refer to `pictures/` folder to have a look at results.

You may have a look at results by running tests scripts which are currently the following :
- `rotate_crop_tests.py`: doing its best to rotate the neck as horizontally as possible and cropping image around the neck
- `grid_detection_tests.py`: working hard on the construction of the grid of notes (i.e. the separation between strings and between frets)
- `finger_detection_tests.py`: concentrating its energy on the detection of fingertips on the neck (but currently failing)

Time performance will be displayed as well as original images and result images.

Should you have a look at how the code is running, open `rotate_crop.py`, `grid_detection.py` and `finger_detection.py`.

## Credits

Here are the papers I had a look at to help me in this project:
- *Vision-Based Guitarist Fingering Tracking using a Bayesian Classifier and Particle Filters* by Kerdvibulvech et al. (2007)
- *Retrieval of Guitarist Fingering Information using Computer Vision* by Scarr et al. (2010)
- *Computer Vision Method for Guitarist Fingering Retrieval* by Burns et al. (2011)

## Contact

paul.de-nonancourt (at) student.ecp.fr
