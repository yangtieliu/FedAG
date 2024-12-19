# FedAG: Enhancing Federated Learning with Adaptive Generators for Personalized Knowledge Distillation

Research code that accompanies the paper FedAG: Enhancing Federated Learning with Adaptive Generators for Personalized Knowledge Distillation.
It contains implementation of the following algorithms:
* **FedAG** (the proposed algorithm) 
* **FedAvg** 
* **FedProx**
* **FedGen**
* **FedFTG** 

## Install Requirements:
```pip3 install -r requirements.txt```

  
## Prepare Dataset: 
* To generate *non-iid* **Mnist** Dataset following the Dirichlet distribution D(&alpha;=0.1) for 20 clients, using 50% of the total available training samples:
<pre><code>cd FedAG/data/Mnist
python generate_niid_dirichlet.py --n_class 10 --sampling_ratio 0.5 --alpha 0.1 --n_user 20
### This will generate a dataset located at FedAG/data/Mnist/u20c10-alpha0.1-ratio0.5/
</code></pre>
    
## Run Experiments: 

There is a main file "main.py" which allows running all experiments.

#### Run experiments on the Dataset:
```
For example:
python main.py --dataset Mnist-alpha0.1-ratio0.5 --algorithm FedGen --batch_size 32 --num_glob_iters 200 --local_epochs 20 --num_users 10 --lamda 1 --learning_rate 0.01 --model cnn --personal_learning_rate 0.01 --times 3 

```
----

### Plot
For the input attribute **algorithms**, list the name of algorithms and separate them by comma, e.g. `--algorithms FedAvg,FedGen,FedProx,FedAG`
```
  python main_plot.py --dataset EMnist-alpha0.1-ratio0.1 --algorithms FedAvg,FedAG --batch_size 32 --local_epochs 20 --num_users 10 --num_glob_iters 200 --plot_legend 1
```
