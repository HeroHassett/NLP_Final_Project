# NLP_Final_Project
This is the final project for group 32 in NLP. 

# Instructions given by prof
You propose to create an evaluation system for clinical named entity recognition (NER) and use this system to evaluate the output of clinical NER systems. You plan to use the RaTE-NER dataset (radiology NER) for evaluation.

You plan to evaluate  two encoder models that have been fine-tuned on the BioClinical-ModernBERT-base and ModernBERT-base datasets. In addition, you will evaluate one decoder model (Llama 4) that has been prompt for inference without any fine-tuning. You may evaluate other models if there is time. All models will be run on the same data and scored with the same code.

You plan to evaluate using precision, recall and F1.

You  may  use five-fold cross validation (CV) on the training and development data,

You may also measure and compare resource usage among systems (GPU, CPU usage, power (watts), etc.

You will also do some error analysis, e.g., using a confusion matrix.