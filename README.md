# meeting_summarization
Parameter-efficient fine-tune with [prompt-tuning](https://arxiv.org/abs/2104.08691) tuned [bigscience/bloomz-3b](https://huggingface.co/bigscience/bloomz-3b) on meeting transcripts.

## Quickstart
### Prerequisites
#### virtualenv option
* Create a python virtual environment `virtualenv venv`
* Source `source venv/bin/activate`

#### conda option
* Create a python virtual environment 
* `conda create --name venv python=3.7`
* Source `conda activate venv`

### Installing
* Install required python package `pip install -r requirements.txt`

### Training
* Download AMI and ICSI meetning transcripts dataset from https://github.com/guokan-shang/ami-and-icsi-corpora 
* run 
```bash
python data_preprocess.py
```
* Place the generated data files in `data` folder.

* Run the following command for model training
```bash
python run_meeting_summarization.py
```
