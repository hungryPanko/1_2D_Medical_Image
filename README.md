# 1_2D_Medical_Image

# _00_Project_Pneumonia_Detection_Chest_Xray

## **Project Overview**

The purpose of this project is to analyze data from the NIH Chest X-ray Dataset and train a CNN to classify a given chest x-ray for the presence or absence of pneumonia. 
This project will culminate in a model that can predict the presence of pneumonia with human radiologist-level accuracy that can be prepared for submission to the FDA for 510(k) clearance as software as a medical device. 
As part of the submission preparation, it will formally describe the model, the data that it was trained on, and a validation plan that meets FDA criteria.

It containsthe medical images with clinical labels for each image that were extracted from their accompanying radiology reports.
The project will include access to a GPU for fast training of deep learning architecture, as well as access to 112,000 chest x-rays with disease labels acquired from 30,000 patients.



## **Project Highlight**

This project is designed to give you hands-on experience with 2D medical imaging data analysis and preparation of a medical imaging model for regulatory approval.

It contians:
- Recommend appropriate imaging modalities for common clinical applications of 2D medical imaging
- Perform exploratory data analysis (EDA) on medical imaging data to inform model training and explain model performance
- Establish the appropriate ‘ground truth’ methodologies for training algorithms to label medical images
- Extract images from a DICOM dataset
- Train common CNN architectures to classify 2D medical images
- Translate outputs of medical imaging models for use by a clinician
- Plan necessary validations to prepare a medical imaging model for regulatory approval




## **Project Steps**

This project has the following steps.

- Exploratory Data Analysis
- Building and Training Your Model
- Clinical Workflow Integration
- FDA Preparation


## **About Files**

- EDA.ipynb: This is the file you will be performing the EDA.
- Build and train model.ipynb: This is the file you will be building and training your model.
- Inference.ipynb: This is the file you will be performing clinical workflow integration.
- .dcm files: They are the test files to test the clinical workflow integration.
- sample_labels.csv: This is the file that should be used to assess images in the pixel-level.
- FDA_Submission_Template.md: This is the template to create the FDA submission. 
