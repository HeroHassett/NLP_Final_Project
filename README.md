# NLP_Final_Project
This is the final project for group 32 in NLP. 

# Instructions given by prof
You propose to create an evaluation system for clinical named entity recognition (NER) and use this system to evaluate the output of clinical NER systems. You plan to use the RaTE-NER dataset (radiology NER) for evaluation.

You plan to evaluate  two encoder models that have been fine-tuned on the BioClinical-ModernBERT-base and ModernBERT-base datasets. In addition, you will evaluate one decoder model (Llama 4) that has been prompt for inference without any fine-tuning. You may evaluate other models if there is time. All models will be run on the same data and scored with the same code.

You plan to evaluate using precision, recall and F1.

You  may  use five-fold cross validation (CV) on the training and development data,

You may also measure and compare resource usage among systems (GPU, CPU usage, power (watts), etc.

You will also do some error analysis, e.g., using a confusion matrix.


# Instructions for running batch-based decoder inference
Command: python decode.py --model_name meta-llama/Llama-4-Scout-17B-16E-Instruct --wandb_project (project name) --batch_size (desired batch size) --hf_token (insert your token)

You can alter batch size and wandb_project
Also you should go into the decode.py file and go to the wandb.init command on line 180 and change the dir param to whatever you prefer

# Instructions for running the sweep
Command: python wandb_sweep.py --model_name meta-llama/Llama-4-Scout-17B-16E-Instruct --project (wandb project name) --sweep_name llama4-decoder-sweep --count 10 --data_dir all --split validation --batch_size (whatever you want) --save_predictions_dir decoder_outputs/sweep_preds --hf_token (insert your hugging face token)

Go to line 88 of the sweep and alter the dir param of the wandb.init command to whatever you want
