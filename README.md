# Computer-Vision

Learning Deep Features for Stain-free Live-dead Human Breast Cancer Cell Classification
Gisela Pattarone1 - Laura Acion2 - Marina Simian3 - Emmanuel Iarussi4

1 Universidad de Buenos Aires, Facultad de Farmacia y Bioquimica, Buenos Aires,
2 Instituto de Cálculo, Facultad de Ciencias Exactas y Naturales, Universidad de Buenos Aires,
3 Instituto de Nanosistemas, Universidad Nacional de San Martín,
4 National Technological University

Abstract
Automated cell classification in cancer biology is a challenging topic in computer vision and machine learning research. Breast cancer is the most common malignancy in women that usually involves phenotypically diverse populations of breast cancer cells and an heterogeneous stroma. In recent years, automated microscopy technologies are allowing the study of live cells over extended periods of time, simplifying the task of compiling large image databases. For instance, there have been several studies oriented towards building machine learning systems capable of automatically classifying images of different cell types (i.e. motor neurons, stem cells). In this work we were interested in classifying breast cancer cells as live or dead, based on a set of automatically retrieved morphological characteristics using image processing techniques. Our hypothesis is that live-dead classification can be performed without any staining and using only bright-field images as input. To our knowledge, there is no previous work attempting this task on in vitro studies of breast cancer cells, nor is there a dataset available to explore solutions related to this issue. We tackled this problem using the JIMT-1 breast cancer cell line that grows as an adherent monolayer. First, a vast image set composed by JIMT-1 human breast cancer cells that had been exposed to a chemotherapeutic drug treatment (doxorubicin and paclitaxel) or vehicle control was compiled. Next, several state-of-the-art classifiers were trained based on convolutional neural networks (CNN) to perform supervised classification using labels obtained from fluorescence microscopy images associated with each bright-field image. Model performances were evaluated and compared on a large number of bright-field images. The best model reached an AUC = 0.941 for classifying breast cancer cells without treatment. Furthermore, it reached AUC = 0.978 when classifying breast cancer cells under drug treatment. Our results highlight the potential of machine learning and computational image analysis to build new diagnosis tools that benefit the biomedical field by reducing cost, time, and stimulating work reproducibility. More importantly, we analyzed the way our classifiers clusterize bright-field images in the learned high-dimensional embedding and linked these groups to salient visual characteristics in live-dead cell biology observed by trained experts.


Our paper on dead and living breast cancer cell image classification was placed as one of the top 100 downloaded cancer papers for Nature Scientific Reports (among more than 1,440 cancer papers in 2021). More info here: https://www.nature.com/collections/gdfhjfggib.

![image](https://user-images.githubusercontent.com/91725761/163794416-6b2592f5-817d-4b33-8122-ef2db6b531ed.png)
![Patches segmentation](https://user-images.githubusercontent.com/91725761/163794193-8cb07bac-561d-46f3-b5e3-37f188d0c741.jpg)
![image](https://user-images.githubusercontent.com/91725761/163794383-653fb0d8-33d3-487f-b789-c07255b0838c.png)

The image dataset and further resources are available in the public github repository:: https://github.com/emmanueliarussi/live-dead-JIMT-1.git
