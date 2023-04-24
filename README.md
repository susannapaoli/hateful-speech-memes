# The Hateful Memes Challenge: Detecting Hate Speech in Multimodal Memes
Using a concatenation-based fusion model, we test different pre-trained large language models to classify hateful content in multimodal memes. The repo is organized in the following way: 

- each language model folder contains a dataloader class that uses the specific tokenizer to pre-process the dataset. We have BERT, GPT2, XLNet and DeHateBert(specific for hate speech recognition)
- 'json data' contains json files for unimodal training (captions)
- 'unimodal experiments' contains notebooks for language model trainings (one for each language model)
- 'multimodal experiments' contains notebooks for fusion model trainings (one for each language model)
- 'resnet' contains class for image model initialization
- 'utils' contains plot curve functions and training/validation functions for unimodal and multimodal models

Language models tested in this analysis are: BERT, XLNet, GPT2, DeHateBert \\
Image model used in this analysis is: RESNET 
