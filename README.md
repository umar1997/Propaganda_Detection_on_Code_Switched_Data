
# Detecting Propaganda Techniques in Code-Switched Social Media Texts  


### ABSTRACT
> Propaganda is a planned persuasive form of communication whose goal is to influence
the opinions and mindset of a target audience or the public in general towards a specific agenda. Propaganda presents a definite tone while expressing its messages, which is a conscious act manifested by a certain individual, group or institution in an attempt to propagate their own narratives. With the advent of the internet and the rise in the number of social media platforms the spread of falsified information and distorted arguments in the form of propaganda has begun to spread on a massive scale. Propaganda can influence individuals by shaping their attitudes and beliefs about certain issues, events, or individuals. It can also influence their behavior, such as how they vote, what products they buy, or even how they view certain groups of people. Most work in propaganda detection has been done on high-resourced languages such as English, Arabic and Spanish. However, little effort has been made to detect propaganda on resource starved languages. Most low-resourced language communities’ resort to mixing multiple languages especially on social media platforms in the form of either post, tweets or comments. This phenomenon of mixing of multiple languages is referred to as code-switching. Code-switching involves switching between two or more languages in a sentence or phrase to express their ideas and thoughts more accurately and convincingly. In general, code-switching brings together high-resourced and low-resourced languages within the same text. To contribute to a healthier online environment, we propose a novel task of detecting propaganda techniques in code-switched data involving English and Roman Urdu. We create a corpus of 1030 code-switched texts which we manually annotate on a fragment-level with 20 propaganda techniques and make it publicly available. For fragment-level annotation on our code-switched texts we develop a web-based annotation platform with an interface that allows easy labelling of spans of text. Furthermore, to do preliminary analysis on our newly created dataset we run experiments using several state-of-the-art pre-trained multilingual and cross-lingual language models namely BERT, mBERT and XLM RoBERTa with different fine-tuning strategies and discover that XLM RoBERTa fine-tuned on Roman Urdu outperforms all other models on our task and dataset. Our work brings a new perspective to the understanding of how propaganda can be detected within the context of multiple languages used in social media communications.

### CONTRIBUTIONS
1. **Formulation of new NLP Task:** We are the first to formulate the new NLP problem of
detecting propaganda techniques in code-switched data.
2. **New Dataset:** We built and annotated a new corpus for this specific problem in the languages
English and Roman Urdu.
3. **Evaluating different NLP Models:** We run various baseline models on our newly created
dataset and evaluate their performance.
4. **Developed a Web-based Platform:** We design and create a new website platform with a
user interface to detect propaganda on text as well as annotate spans of text and label them
as propaganda techniques.

### FOLDERS
**DATASET** 
> In this folder we have our annotated code-switched dataset as **Dataset.csv** and **dataset.json**
> The code to run the dataset statistics is in **dataset_statistics.ipynb**
> The subfolder **Annotators_Training/** contains the training files used to train the annotators 


**MODELS** 
> In this folder we have the code where we run our various baseline models for multi-label sentence classification on our code-switched dataset
> The token classification model code we run for our fragment-level task on the meme text-only data is in the folder **Meme_Files/** (We use this model for our Propaganda App as well)
> The training, validation and test split for the code-switched text and the meme text-only data is in folder **Data_Files/**

**PROPAGANDA APP** 
> In this folder is the code for the website we create for annotating our code-switched text on a fragment-level to 20 propaganda techniques/classes.
> We create our website using the Flask framework and HTML, CSS and AJAX
> The website also runs a model to detect spans of text as propaganda on a fragment-level

### Contact
Should you have any question, please contact umar.salman1997@gmail.com