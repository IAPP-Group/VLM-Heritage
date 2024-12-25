# A Comparative Study of Vision Language Models for Italian Cultural Heritage
Chiara Vitaloni *, Dasara Shullani, Daniele Baracchi


Manuscript under review on *Heritage* in the Special Issue: *AI and the Future of Cultural Heritage*.

## Abstract

 Human communication has long relied on images for both active and passive interaction. For several decades now, electronic devices equipped with a screen have been used to search for and obtain visual data. Until recently, however, the flow of visual information was unidirectional, as input queries needed to be text fragments. At the same time, improvements in human-computer interaction technologies made it possible to query search engines such as Google using visual data (a technique known as “reverse image search”). In recent times, technologies like large language models have brought together these two approaches, enabling the inclusion of both textual questions and images within a single query. These tools have been explored in part by the scientific community for cultural heritage-related applications such as searching for information on artworks. In this context, this paper investigates the use of a wide range of Vision-Language Models (VLMs), including Bing’s search engine with GPT-4 and open models like Qwen2-VL and Pixtral, for cultural heritage visual question answering. To do so, twenty subjects were chosen to represent well-known Italian landmarks (i.e. Colosseo, Milan Cathedral, Michelangelo’s David in Florence). For each subject, two pictures were selected: one from Wikipedia and one either from a scientific database or from private collections of pictures. These images were input into each VLM alongside textual queries about their content. We studied the quality of the responses in terms of their completeness, assessing the impact of various levels of detail in the queries. Additionally, we evaluated the impact of language (English or Italian) on the system’s ability to provide satisfactory answers.
 
*Keywords*: visual question answering; cultural heritage; artificial intelligence; ChatGPT; human-centered approaches



## Folder Organization

- `dataset` contains all the images used in the anaylsis
- `results` contains the responses in ITA/ENG provided by each open VLM
- `dataset-eval` contains the responses and the evaluation
- `final-results` contains the evaluation of all algorithms in Italian and in English 

