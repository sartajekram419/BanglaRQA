# BanglaRQA

This repository contains the official release of the dataset "BanglaRQA" and implemented model codes of the paper titled [**"BanglaRQA: A Benchmark Dataset for Under-resourced Bangla Language Reading Comprehension-based Question Answering with Diverse Question-Answer Types"**](https://aclanthology.org/2022.findings-emnlp.186) accpeted in *Findings of EMNLP 2022*.


## Table of Contents

- [Dataset](#dataset)
- [Codes](#codes)
- [Models](#models)
- [Results](#results)
- [License](#license)
- [Citation](#citation)

## Dataset

We have released the Bangla Question-Answering dataset 'BanglaRQA' introduced in the paper.
- [**BanglaRQA**](https://huggingface.co/datasets//)

## Codes

The codes for training and testing the models is provided.

## Models

The finetuned model checkpoints are available at [Huggingface model hub](https://huggingface.co/amaderaccount).

- [**finetuned-BanglaT5-for-BanglaRQA**](https://huggingface.co/csebuetnlp/amadermodel)
- [**finetuned-mT5-for-BanglaRQA**](https://huggingface.co/csebuetnlp/amadermodel)
- [**finetuned-BanglaBERT-for-BanglaRQA**](https://huggingface.co/csebuetnlp/amadermodel)
- [**finetuned-mBERT-for-BanglaRQA**](https://huggingface.co/csebuetnlp/amadermodel)
  
To use these models for the supported downstream tasks in this repository see **[Training & Evaluation](#training--evaluation).**

***Note:*** These models were pretrained using a ***specific normalization pipeline*** available **[here](https://github.com/csebuetnlp/normalizer)**. All finetuning scripts in this repository uses this normalization by default. If you need to adapt the pretrained model for a different task make sure ***the text units are normalized using this pipeline before tokenizing*** to get best results. A basic example is available at the **[model page](https://huggingface.co/csebuetnlp/banglabert).**



## Testing

To use the pretrained model for finetuning / inference on different downstream tasks see the following section:

* **[Sequence Classification](sequence_classification/).**
  - For single sequence classification such as
    - Document classification
    - Sentiment classification
    - Emotion classification etc.
  - For double sequence classification such as 
    - Natural Language Inference (NLI)
    - Paraphrase detection etc.
- **[Token Classification](token_classification/).**
  - For token tagging / classification tasks such as
    - Named Entity Recognition (NER)
    - Parts of Speech Tagging (PoS) etc.
- **[Question Answering](question_answering/).**
    - For tasks such as,
      - Extractive Question Answering
      - Open-domain Question Answering


## Benchmarks
 
* Zero-shot cross-lingual transfer-learning

|     Model          |   Params   |     SC (macro-F1)     |      NLI (accuracy)     |    NER  (micro-F1)   |   QA (EM/F1)   |   BangLUE score |
|----------------|-----------|-----------|-----------|-----------|-----------|-----------|
|[mBERT](https://huggingface.co/bert-base-multilingual-cased) | 180M  | 27.05 | 62.22 | 39.27 | 59.01/64.18 |  50.35 |
|[XLM-R (base)](https://huggingface.co/xlm-roberta-base) |  270M   | 42.03 | 72.18 | 45.37 | 55.03/61.83 |  55.29 |
|[XLM-R (large)](https://huggingface.co/xlm-roberta-large) | 550M  | 49.49 | 78.13 | 56.48 | 71.13/77.70 |  66.59 |
|[BanglishBERT](https://huggingface.co/csebuetnlp/banglishbert) | 110M | 48.39 | 75.26 | 55.56 | 72.87/78.63 | 66.14 |

* Supervised fine-tuning

|     Model          |   Params   |     SC (macro-F1)     |      NLI (accuracy)     |    NER  (micro-F1)   |   QA (EM/F1)   |   BangLUE score |
|----------------|-----------|-----------|-----------|-----------|-----------|-----------|
|[mBERT](https://huggingface.co/bert-base-multilingual-cased) | 180M  | 67.59 | 75.13 | 68.97 | 67.12/72.64 | 70.29 |
|[XLM-R (base)](https://huggingface.co/xlm-roberta-base) |  270M   | 69.54 | 78.46 | 73.32 | 68.09/74.27  | 72.82 |        
|[XLM-R (large)](https://huggingface.co/xlm-roberta-large) | 550M  | 70.97 | 82.40 | 78.39 | 73.15/79.06 | 76.79 |
|[sahajBERT](https://huggingface.co/neuropark/sahajBERT) | 18M | 71.12 | 76.92 | 70.94 | 65.48/70.69 | 71.03 |
|[BanglishBERT](https://huggingface.co/csebuetnlp/banglishbert) | 110M | 70.61 | 80.95 | 76.28 | 72.43/78.40 | 75.73 |
|[BanglaBERT](https://huggingface.co/csebuetnlp/banglabert) | 110M | 72.89 | 82.80 | 77.78 | 72.63/79.34 | **77.09** |


The benchmarking datasets are as follows:
* **SC:** **[Sentiment Classification](https://aclanthology.org/2021.findings-emnlp.278)**
* **NER:** **[Named Entity Recognition](https://multiconer.github.io/competition)**
* **NLI:** **[Natural Language Inference](#datasets)**
* **QA:** **[Question Answering](#datasets)**


## License
Contents of this repository are restricted to non-commercial research purposes only under the [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License (CC BY-NC-SA 4.0)](https://creativecommons.org/licenses/by-nc-sa/4.0/). 

<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/4.0/88x31.png" /></a>

## Citation
If you use the dataset, or any of the models or code modules, please cite the following paper:
```
@inproceedings{ekram-etal-2022-banglarqa,
    title = "{B}angla{RQA}: A Benchmark Dataset for Under-resourced {B}angla Language Reading Comprehension-based Question Answering with Diverse Question-Answer Types",
    author = "Ekram, Syed Mohammed Sartaj  and
      Rahman, Adham Arik  and
      Altaf, Md. Sajid  and
      Islam, Mohammed Saidul  and
      Rahman, Mehrab Mustafy  and
      Rahman, Md Mezbaur  and
      Hossain, Md Azam  and
      Kamal, Abu Raihan Mostofa",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2022",
    month = dec,
    year = "2022",
    address = "Abu Dhabi, United Arab Emirates",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.findings-emnlp.186",
    pages = "2518--2532",
    abstract = "High-resource languages, such as English, have access to a plethora of datasets with various question-answer types resembling real-world reading comprehension. However, there is a severe lack of diverse and comprehensive question-answering datasets in under-resourced languages like Bangla. The ones available are either translated versions of English datasets with a niche answer format or created by human annotations focusing on a specific domain, question type, or answer type. To address these limitations, this paper introduces BanglaRQA, a reading comprehension-based Bangla question-answering dataset with various question-answer types. BanglaRQA consists of 3,000 context passages and 14,889 question-answer pairs created from those passages. The dataset comprises answerable and unanswerable questions covering four unique categories of questions and three types of answers. In addition, this paper also implemented four different Transformer models for question-answering on the proposed dataset. The best-performing model achieved an overall 62.42{\%} EM and 78.11{\%} F1 score. However, detailed analyses showed that the performance varies across question-answer types, leaving room for substantial improvement of the model performance. Furthermore, we demonstrated the effectiveness of BanglaRQA as a training resource by showing strong results on the bn{\_}squad dataset. Therefore, BanglaRQA has the potential to contribute to the advancement of future research by enhancing the capability of language models. The dataset and codes are available at https://github.com/sartajekram419/BanglaRQA",
}


```
