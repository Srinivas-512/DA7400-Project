To run the model, there are two steps:

1) Train the CVAE 

2) Train and evaluate the low level agent in conjuction with the High Level Planner


To complete step 1), the following steps must be taken:
    - change working directory to the CVAE folder (cd CVAE)
    - open train.py and modify the object creation to include:
        i) Latent and Hidden dimensions of choice
        ii) Number of epochs to train
        iii) The environment on which to train the CVAE
    - run train.py (python train.py)

This will automatically save the weights as well with naming including the env, latent dim, hidden dim


For step 2), the following are the steps to be taken:
    - change working directory to higoc folder (cd higoc) 
    - open config.py and change any hyperparams as required for either of the two agents
    - run => python higoc.py --agent_type 'iql' --exp_name 'exp0' --env 'antmaze-umaze-v2'  --latent_dim 8 --hidden_dim 64 --subgoal_patience 50
        i) change the CVAE parameters appropriately to values on which the CVAE has been trained before,
           this will automatically trigger the collection of those weights
        ii) change the name of the experiment for different runs else the weights saved will be
            overwritten
        iii) agent type has two choices (iql or td3_bc)

This will automatically save weights as well in the folder weights/{agent_type}-{exp_name} in the higoc itself