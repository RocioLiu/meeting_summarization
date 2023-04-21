# meeting_summarization
Implementing BERT + CRF with PyTorch for Chinese named entity recognition (NER).
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
* Download People’s Daily dataset from https://github.com/OYE93/Chinese-NLP-Corpus/tree/master/NER and place the data files in `data` folder.
* Run the following command
```bash
python run_ner.py
```
