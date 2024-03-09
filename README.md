# Image Captioning for Remote Sensing Data
Team Name : TRINIT_giant-spoons_ML

[Demo Video](https://drive.google.com/file/d/15pppNEg9lJwh0UlrcC6isNwk925E4yTJ/view?usp=sharing)



## Introduction
This project involves creation of an application for image captioning remote sensing data (satellite imagery) from the RSICD Dataset.

From urban planning and environmental monitoring to disaster management and agricultural analysis, the applications of remote sensing data are diverse and far-reaching.

## Dataset Description
The dataset consists of three primary files: train.csv, test.csv, and valid.csv. These files
contain information about image filenames and their respective captions. Each file includes multiple
captions for each image to support diverse training techniques.

- `train.csv`: This file contains filenames (filename column) and their corresponding captions
(captions column) for training your image captioning model.
- `test.csv`: The test set is included in this file, which contains a similar structure as that of
train.csv. The purpose of this file is to evaluate your trained models on unseen data.
- `valid.csv`: This validation set provides images with their respective filenames (filename)
and captions (captions). It allows you to fine-tune your models based on performance
during evaluation.

## Evaluation Metric
**BLEU (Bilingual Evaluation Understudy) Score** - BLEU score provides a quantitative measure of the quality of generated captions compared to reference captions. 

## Model Used
**GIT (GenerativeImage2Text), base-sized**:
- GIT (short for GenerativeImage2Text) model, base-sized version. It was introduced in the paper [GIT: A Generative Image-to-text Transformer for Vision and Language](https://arxiv.org/abs/2205.14100) by Wang et al.

### Model description
- GIT is a Transformer decoder conditioned on both CLIP image tokens and text tokens. The model is trained using "teacher forcing" on a lot of (image, text) pairs.
- The goal for the model is simply to predict the next text token, giving the image tokens and previous text tokens.
- The model has full access to (i.e. a bidirectional attention mask is used for) the image patch tokens, but only has access to the previous text tokens (i.e. a causal attention mask is used for the text tokens) when predicting the next text token.
- We fine-tuned the GIT-Base Model on the RSCID Dataset.

## Application
The streamlit application involves taking the image as input from the user and getting inference from the model as the generated caption, as seen in the demonstration.