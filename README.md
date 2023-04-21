# meeting_summarization
Parameter-efficient fine-tuning with [prompt-tuning](https://arxiv.org/abs/2104.08691) tuned [bigscience/bloomz-3b](https://huggingface.co/bigscience/bloomz-3b) on meeting transcripts summarization.

## Quickstart
### Prerequisites
#### virtualenv option
* Create a python virtual environment `virtualenv venv`
* Source `source venv/bin/activate`

#### conda option
* Create a python virtual environment 
* `conda create --name venv python=3.9`
* Source `conda activate venv`

### Installing
* Install required python package `pip install -r requirements.txt`

### Data preprocess
* Download AMI and ICSI meetning transcripts dataset from https://github.com/guokan-shang/ami-and-icsi-corpora 
* run the following command from data preprocessing
```bash
python src/data_preprocess.py
```
* Place the generated data files in `data` folder.

### Training
* Run the following command for model training
```bash
python src/run_meeting_summarization.py
```
