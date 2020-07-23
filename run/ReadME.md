##The environment has been setup in GCP project image2model with instance p-ds-1. Associated conda environment: detetron2##

To setup:
* Create conda environment 
* Install all associated files from:
* pip install -r requirements.txt
* Install CUDA(recomended 10.2 for detectron2): Can be a bit tricky. refer to my doc to install in GCP: 
   * [https://docs.google.com/document/d/1zuymbahC4TVNKIxTQM8PLdZv4P2kWsDtjiYYtfPXdU8/edit?usp=sharing]
* Install Pytorch, torchvision.[https://pytorch.org/]
    * Current GCP runs on: conda install pytorch torchvision cudatoolkit=10.2 -c pytorch
* install detectron2 [https://colab.research.google.com/drive/16jcaJoc6bCFAQ96jDe2HwtXj7BMD_-m5]

Run
* Run **run-detectron2.py 'input-image' 'output-image'** (Gives segmented image)
    * Eg: run-detectron2.py sample-img/Alondria+Mesh+Task+Chair.jpg sample-img/outp.png
* Run **transferlearn.py <input-image=output image of previous step>** (Needs: folder with structure subclass-check: has synthetic data of all subclasses)
    * Eg: transferlearn.py sample-img/outp.png
    * Ref1 [https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html]
* Final output: class of image (90% accuracy), subclass of image based on image-feature based similarity score 
    * Ref1 [https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_orb/py_orb.html#orb]
    * Ref2 [https://gist.github.com/duhaime/211365edaddf7ff89c0a36d9f3f7956c]
