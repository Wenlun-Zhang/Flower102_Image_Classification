# Flower102_Image_Classification
 Image classification task for <102 Category Flower Dataset>  
 Dataset from: https://www.robots.ox.ac.uk/~vgg/data/flowers/102/


 ## bins: One-shot script
  -->> split_flower_dataset.py: Divide images into train/valid/test set.  
  -->> model_evaluation.py: Inference-only script on test set to evaluate the model.  
  -->> parse_misclassification.py: Parsing misclassification classes with misclassified images picked up.  
  -->> reorder_flower_dataset.py: Reorder the dataset with paired image and label.

 ## tools: Some useful tools
  -->> common_tools.py: Useful tools.  
  -->> model_trainer.py: Model trainer.  
  -->> load_model.py: Model select, modify this when using specified model.

 ## dataset: Dataset
  -->> flower_102.py: Dataset.

 ## config: Configuration management
  -->> flower_config.py: Parameter config, modify this before training.

 ## src: Source code
  -->> flower_train.py: Main source code.

 ## run: Export results and log file in this folder
