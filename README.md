# cell_segmentation
## Introduction
Microscopy derived live-cell-painting image sequences taken by the Kallioniemi-group using high-resolution high-throughput microscopy will be frame-by-frame segmented using Deep Learning methods. Figure 1 shows the different types of segmentation, for example, image classification, object detection, semantic detection, and instance segmentation. In this project, instance segmentation is chosen to be the segmentation goal. Instance segmentation can classify each class and also each instance in the image which is useful for cell imaging. The cell can be segmented one by one and the segmented cell image can be used for further analysis. Mask R-CNN architecture is used to build the instance segmentation model in Pytorch.


![cell_segmentation_result](https://user-images.githubusercontent.com/79293659/209575581-b505b735-ff66-4c6b-9032-01bf9d0b36fe.png)
