# Dependencies
* Pytorch 0.4.1
* numpy
* Python3
* Pillow
* opencv-python
* tenseorboardX
* PyQt5
* pyyaml

# Testing & Color visualization
* Run g.partition.py to segregate test and train labels
* Run `python tester.py`
* Results will be saved in `./test_results`
* Color visualized results will be saved in `./test_color_visualize`

# Gui
* Run `python demo.py`
* Gui will be openend
* Insert mask using open mask button and insert image using Open image button in GUI
* click on the buttons provided such as hair, eyeglassess, neck, skin..... to edit the segmentation mask 
* click on edit to see the manipulated image
* save the image 

# Evaluation metric for same masks and images
* Run `python Evaluation.py`
* pip install lpips
* pip install torch torchvision


# Evaluation metric for different masks and images
* Run `python Evaluation_cross_masks.py`


