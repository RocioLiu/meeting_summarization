from os.path import abspath, dirname, split, join, basename
import glob
import json
import re
from typing import Dict, List


ROOT_PATH = join(*split(abspath(dirname("__file__"))))
DATA_PATH = join(ROOT_PATH, "data")

id2role = {"PM": "project manager", 
           "UI": "user interface", 
           "UX": "user experience",
           "ID": "industrial designer",
           "ME": "marketing expert",
           "Professor": "professor",
           "Postdoc": "Postdoc",
           "PhD": "PhD",
           "Grad": "graduate student",
           "Undergrad": "undergraduate student",
           }

# data cleaning with removing `<>` and redundancy
pattern_list = [r"\<[A-Za-z0-9_]+\>"]
words_remove_list = ["Mm", "mm", "um", "Um", "un", "Un", "uh", "Uh", "hmm"]

# ami train / valid / test split by id
ami_train = ['ES2002', 'ES2005', 'ES2006', 'ES2007', 'ES2008', 'ES2009', 'ES2010', 'ES2012', 'ES2013', 'ES2015', 'ES2016', 'IS1000', 'IS1001', 'IS1002', 'IS1003', 'IS1004', 'IS1005', 'IS1006', 'IS1007', 'TS3005', 'TS3008', 'TS3009', 'TS3010', 'TS3011', 'TS3012']
ami_validation = ['ES2003', 'ES2011', 'IS1008', 'TS3004', 'TS3006']
ami_test = ['IS1009', 'TS3007']

# icsi train / valid / test split by id
icsi_validation = ['Bed004', 'Bed009', 'Bed016', 'Bmr019']
icsi_test = ['Bmr005','Bro018']


def regex_remove(text, pattern_list):
    for pattern in pattern_list:
        text = re.sub(pattern, '', text)
    return text

def multiple_word_remove(text, words_remove_list):
    words = text.split(" ")
    text = ' '.join([word for word in words if word not in words_remove_list])
    return text

def train_test_split_flag(transcript_id, valid_idx, test_idx):
    # if the last character of transcript_id is a lowercase letter, 
    # remove it in order to match the split index
    transcript_id = transcript_id[:-1] if bool(re.search(r"[a-z]", transcript_id[-1])) else transcript_id
    if transcript_id in valid_idx:
        data_split = "valid"
    elif transcript_id in test_idx:
        data_split = "test"
    else:
        data_split = "train"
    return data_split


def get_summary(row, summary) -> str:
    if len(summary) == 0:
        concact_symbol = ""
    else:
        concact_symbol = " "
    summary += concact_symbol + row["abstractive"]["text"]
    return summary

def get_paragraph_summary(row) -> str:
    summary = row["abstractive"]["text"]
    return summary

def get_transcript(row, transcript, corpus_name) -> str:
    for j in range(len(row["extractive"])):
        role_id = row['extractive'][j]['attributes']['role']
        role_name = id2role[role_id]
        if corpus_name == "icsi":
            role = role_name + " " + row['extractive'][j]['speaker']
        else:
            role = role_name
        text = row['extractive'][j]['text']
        text = regex_remove(text, pattern_list)
        text = multiple_word_remove(text, words_remove_list)       

        if len(transcript) == 0:
            transcript += f"Speaker {role}: {text}"  
        elif j == 0:
            transcript += f"\nSpeaker {role}: {text}"      
        elif role == previous_role:
            transcript += f". {text}" if not bool(re.search(r"\.", transcript[-1])) else f" {text}"
        else:
            transcript += f"\nSpeaker {role}: {text}"
        previous_role = role

    return transcript

def get_paragraph_transcript(row, corpus_name):
    for j in range(len(row["extractive"])):
        role_id = row['extractive'][j]['attributes']['role']
        role_name = id2role[role_id]
        if corpus_name == "icsi":
            role = role_name + " " + row['extractive'][j]['speaker']
        else:
            role = role_name
        text = row['extractive'][j]['text']
        text = regex_remove(text, pattern_list)
        text = multiple_word_remove(text, words_remove_list)       
 
        if j == 0:
            transcript = f"Speaker {role}: {text}"      
        elif role == previous_role:
            transcript += f". {text}" if not bool(re.search(r"\.", transcript[-1])) else f" {text}"
        else:
            transcript += f"\nSpeaker {role}: {text}"
        previous_role = role

    return transcript

def preprocess_a_transcript(transcript_id, data: List, corpus_name):
    """
    params:
        data: a list of one raw summlink
    return:
        summary_list: a list of paragraph's summary
    """
    summary_list = []
    transcript_list = []
    for i in range(len(data)):
        # each row is an ('abstractive', 'extractive') pair (i.e. a paragraph) of a summlink
        row = data[i]
        if row["abstractive"]["type"] == "abstract":
            summary = get_paragraph_summary(row)
            summary_list.append(summary)
            transcript = get_paragraph_transcript(row, corpus_name)
            transcript_list.append(transcript)
        else:
            break
    return summary_list, transcript_list


def create_summary_file(corpus_path, output_name, pattern_list, words_remove_list, valid_idx, test_idx):
    """
    params:
        corpus_folder: a folder filled with raw summlink json file
    """
    corpus_name = corpus_path.split("/")[-1].split("-")[-1]
    # get all summlinks as a list
    file_path_list = [t for t in sorted(glob.glob(join(corpus_path, "*")))]

    data_list = []
    global_count = 0
    for i in range(len(file_path_list)):
        # a file_path store the content of a transcript
        file_path = file_path_list[i]
        with open(file_path, "r") as f:
            a_summlink = json.load(f)

        # get the transcript and summary of each summlink, store as a dict, 
        # and then append to data_list
        transcript_id = basename(file_path).split(".")[0]
        summary_list, transcript_list = preprocess_a_transcript(transcript_id, a_summlink, corpus_name)
        data_split = train_test_split_flag(transcript_id, valid_idx, test_idx)
        for j in range(len(summary_list)):
            global_count += 1
            data_list.append({"uid": f"{global_count}-{transcript_id}", "id": transcript_id, "text": transcript_list[j], 
                         "summary": summary_list[j], "split": data_split})

    # write the data to a json file
    with open(join(DATA_PATH, f"{output_name}_{corpus_name}.json"), "w", encoding="utf8") as fp:
        json.dump(data_list, fp, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    create_summary_file(f"{DATA_PATH}/summlink-ami", "meeting_summary", pattern_list, words_remove_list, ami_validation, ami_test)
    create_summary_file(f"{DATA_PATH}/summlink-icsi", "meeting_summary", pattern_list, words_remove_list, icsi_validation, icsi_test)

    
