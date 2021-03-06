**gensynth-blender.py: Uses blender script to generate synthetic data from OBJ files.**

* OBJ files should be in format:
  * Input folder
      * Class 1
        * Class 1 subclass 1
        * Class 1 subclass 2
        * ...
      * Class 2
        * .....
        * ...
      * Class n
        * Class n subclass 1
    
* Assigns random texture to the object
    * picks texture from folder 'texture/'
* Rotates camera in a dolly around the object taking screenshots
* Stores screenshot for each model inside a folder with suffix: '-synthetic_img'

**post-synth.py: post process files generated by blender**
* Resize to image size required by machine learning model
* Remove background to white
* Split data into training and validation (default: 80-20 split)

**prelim-train-vgg.ipynb: Trains fully connected (final) layers of transfer learning model. Output: .h5 file contained weights of trained model**
Ref: [https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html]
Does keras weighted balance based on data skew

**vgg-fine-tuning.ipynb: Loads weights trained in the previous layer and fine tunes entire model. Output: model to be used in 'run'**

**hyper-param-greedy.ipynb: Grid search based hyper parameter optimisation for improvement of accuracy in prelim-train-vgg.ipynb**
