# image2meshinelearning #
## Find 3D mesh correspoding to real image. End to end system ##
Conservative idea: given image find closest mesh
Optimistic idea: Given image, generate 3d mesh and trace if this mesh is part of some other larger object

Run:
Entire pipeline with pre-trained transfer learning module.
* Takes input an image belonging to furniture class. 
* Finds, segments and extracts the furniture from cluttered backgroud. [Uses Detectron 2](https://github.com/facebookresearch/detectron2)
* Runs through pre-trained machine learning architecture purely trianed on synthetic data generated from CAD models.
* Outputs image class (desk, chair, sofa, table)
* Finds matching score with sub-class (eg for chair: recliner, loveseat etc) using computer vision (orb feature) and return nearest CAD model
![Workflow](https://github.com/Smandal-physna/image2meshinelearning/blob/master/workflow.png "Flowchart")
