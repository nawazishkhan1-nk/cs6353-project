# CS-6353 Project
## Nawazish Khan, Sunjoo Lee, Yosuke Mizutani

To run the model, first download [BraTS2020-Dataset](https://www.kaggle.com/datasets/awsaf49/brats20-dataset-training-validation) at this location *./data/* and unzip the downloaded data

Run the *run_model.py* scripts with appropriate flags to train, evaluate or do both.

Relevant Flags: [-model_type, -train, -evaluate]

### Usage Examples

- For the baseline model:
    - Evaluate only: *python run_model.py -model_type baseline -evaluate* 
    - Train and evaluate: *python run_model.py -model_type baseline -train -evaluate*

- For the proposed Cascaded Network:
    - Evaluate only: *python run_model.py -model_type cascaded -evaluate*
    - Train and evaluate: *python run_model.py -model_type cascaded -train -evaluate*
 
Analaysis plots for the report are done separately in the jupyter notebooks.