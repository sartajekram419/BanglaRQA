# BanglaRQA

This repository contains the official release of the dataset "BanglaRQA" and implemented model codes of the paper titled [**"BanglaRQA: A Benchmark Dataset for Under-resourced Bangla Language Reading Comprehension-based Question Answering with Diverse Question-Answer Types"**](https://aclanthology.org/2022.findings-emnlp.186) accpeted in *Findings of the Association for Computational Linguistics: EMNLP 2022*.


## Table of Contents

- [Dataset](#dataset)
- [Codes](#codes)
- [Results](#results)
- [License](#license)
- [Citation](#citation)

## Dataset

We have released the Bangla Question-Answering dataset 'BanglaRQA' introduced in the paper.
- [**BanglaRQA**](https://huggingface.co/datasets/sartajekram/BanglaRQA)

## Codes

The code files for training and testing mBERT, mT5, BanglaBERT and BanglaT5 models are provided.

## Results

Supervised fine-tuning on BanglaRQA for question-answering

|     Model      |   EM/F1   |
|----------------|-----------|
|mBERT | 28.53/39.40 |
|BanglaBERT | 47.55/63.15  |        
|mT5 | 53.52/68.83 |
|BanglaT5 | **62.42/78.11** |



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
