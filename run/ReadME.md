The environment has been setup in GCP project image2model with instance p-ds-1. Associated conda environment: detetron2

To setup:
Create conda environment 
Install all associated files from:
pip install -r requirements.txt
install detectron2 [https://colab.research.google.com/drive/16jcaJoc6bCFAQ96jDe2HwtXj7BMD_-m5]

Run
* Run run-detectron2.py <input-image> <output-image> (Gives segmented image)
* Run transferlearn.py <input-image=output image of previous step> (Needs: folder with structure subclass-check: has synthetic data of all subclasses)
  * Ref1 [https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html]
* Final output: class of image (90% accuracy), subclass of image based on image-feature based similarity score 
  * Ref1 [https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_orb/py_orb.html#orb]
  * Ref2 [https://gist.github.com/duhaime/211365edaddf7ff89c0a36d9f3f7956c]
