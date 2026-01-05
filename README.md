# BioACE


Please install [Anaconda](https://www.anaconda.com/distribution/) to create a conda environment as follows:
```shell script
# preparing environment
conda create -n bioace python=3.10
conda activate bioace
pip install -r requirements.txt
```


## Install Java Dependency
```shell script
wget https://github.com/adoptium/temurin21-binaries/releases/download/jdk-21.0.1+12/OpenJDK21U-jdk_x64_linux_hotspot_21.0.1_12.tar.gz
mkdir -p $HOME/jdk
tar -xzf OpenJDK21U-jdk_x64_linux_hotspot_21.0.1_12.tar.gz -C $HOME/jdk
export JAVA_HOME="$HOME/jdk/jdk-21.0.1+12"
export PATH="$JAVA_HOME/bin:$PATH"
```

## Install Pyserini Dependency
```shell script
conda install -c pytorch faiss-cpu -y
```


## Download data and models
```shell script
./download_data_models.sh
```
It will download required resources in the resource directory.

## Download Llama3.3 
Before downloading, you need to agree to Meta's license terms by visiting here: https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct

You may need to fill out the form to agree to the license terms. Once your request approved, run the following:


```shell script
  pip install huggingface_hub
  huggingface-cli login
  huggingface-cli download meta-llama/Llama-3.3-70B-Instruct --local-dir resources/models/completeness_model

```
The required data format is provided in the file: ```resources/data/task_b_baseline_output.json``` and the ground-truth nuggets format is provide in ```resources/data/task_b_gt_nuggets_first2.json``` 
## Run BioACE for Answer Evaluation
```
export JAVA_HOME="$HOME/jdk/jdk-21.0.1+12"
export PATH="$JAVA_HOME/bin:$PATH"
 cd src/
 python answer_eval.py
```
## Run BioACE for Citation Evaluation
```
 cd src/
 python citation_eval.py
```

