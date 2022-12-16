# CS-6353 Project
## Nawazish Khan, Sunjoo Lee, Yosuke Mizutani

To run the model, first download [BraTS2020-Dataset](https://www.kaggle.com/datasets/awsaf49/brats20-dataset-training-validation) at this location *./data/* and unzip the downloaded data

Run the *run_model.py* scripts with appropriate flags to train, evaluate or do both.

Relevant Flags: [-model_type, -train, -evaluate]

### Usage Examples

- For the baseline model:
    - *python run_model.py -model_type baseline -evaluate*
    - *python run_model.py -model_type baseline -train -evaluate*

- For the proposed Cascaded Network:
    - *python run_model.py -model_type cascaded -evaluate*
    - *python run_model.py -model_type cascaded -train -evaluate*
 