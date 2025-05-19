<p align="center">
  <img src="https://www.merchandisingplaza.co.uk/282130/2/Stickers-Star-Trek-STAR-TREK-Spock-Live-Long-Prosper-Sticker-l.jpg" 
       alt="Live Long and Prosper Sticker" 
       width="300">
</p>

# HW1 Retrieval-based Chat Bot

**Task**: Develop a retrieval-based chat bot using the retrieval-based approach. The bot should engage in dialogue as a specific character from a TV series, imitating their style and manner of speech. It is important to consider the character’s speech patterns, themes they raise, and their typical reactions.

## Data Collection
As a foundation for the chatbot, I used scripts from the *Star Trek* series, which I downloaded from this [repository](https://github.com/varenc/star_trek_transcript_search), specifically the lines of Mr. Spock, a crew member and scientist from the planet Vulcan.

The data was processed as follows:
- script cleaning  
- selecting the character’s lines as possible bot responses  
- extracting the preceding line as the “question”; if it’s the first line of the scene, this field is empty  
- extracting the previous dialogue lines as context; again, if it's the first line, the context is empty. Context = consecutive previous lines  

After processing, about 5,800 character lines were available for the bot to use in conversations. I then prepared datasets for training the candidate ranking and re-ranking models.

Code is in the notebook [GNLP_HW1-data_prep.ipynb](https://github.com/greatakela/ChatBot/blob/main/Notebooks/GNLP_HW1-data_prep.ipynb)

## Training Data for the Bi-Encoder
Based on the processed data, I prepared training samples for a bi-encoder model. Since I used triplet loss, the data is structured into triples:
- **ANCHOR** – context + question
- **ANSWER** – correct response from the script
- **WRONG_ANSWER** – randomly selected line (from the *House M.D.* series)

## Data for Reranker Training
These same datasets are used for training a reranker model to re-rank response candidates. Correct responses are labeled `0` and augmented with lines from other shows as negative examples labeled `1`.

Reranker data includes context-question-answer sequences separated by a special `[SEP]` token. Around 10,000 samples were created, with a 50/50 class split.

## Chat Bot Architecture

The chatbot workflow is shown schematically below.

![image](https://github.com/greatakela/ChatBot/blob/main/static/ArchBot.png)

The **reply database** consists of vectorized scripts using a model from the [trained encoder](https://huggingface.co/greatakela/gnlp_hw1_encoder), including both context and question.

Reply selection happens in two steps:
1. Retrieval of similar context-question pairs based on cosine similarity from the vector database. The top candidates are selected based on similarity to the user’s input.
2. A reranker model then classifies whether the retrieved answer is a valid continuation. Only responses labeled as class `0` (logical continuation) are kept and ranked by the model’s confidence. If all responses are class `1`, the top cosine-similar response is returned.

The **intent classifier** is taken from the [DialogTag](https://pypi.org/project/DialogTag/) library and is used to label both the training data and incoming user inputs. Intent is also used to filter candidates and is embedded as an additional feature in the bi-encoder.

The **bi-encoder model** is based on `distilroberta-base`, fine-tuned on the triplet data using the `sentence-transformers` library. Training uses a **Triplet Loss Function**, minimizing the distance between anchor and correct answer, and maximizing it between anchor and incorrect answer.

The model was evaluated by accuracy—specifically, whether the similarity between anchor and correct answer was higher than with the incorrect one. The untrained `distilroberta-base` achieved 58%, while the trained model reached 98%.

<p align="center"> <img src="https://github.com/greatakela/ChatBot/blob/main/static/evaluator_val.PNG" width="49.5%"> </p>

Training code is in this [notebook](https://github.com/greatakela/ChatBot/blob/main/Notebooks/GNLP_HW1-bi_encoder_model_train.ipynb). The model is hosted on Hugging Face ([link](https://huggingface.co/greatakela/gnlp_hw1_encoder)) and used for inference.

The **reranker model** is based on `bert-base-uncased`, trained on the previously prepared labeled data. Classification performance was evaluated by accuracy. Training results are shown below.

<img src="https://github.com/greatakela/ChatBot/blob/main/static/W%26B%20Chart%203_11_2025%2C%202_38_44%20PM.png" width="49.5%"> <img src="https://github.com/greatakela/ChatBot/blob/main/static/W%26B%20Chart%203_11_2025%2C%202_39_05%20PM.png" width="49.5%">

The model achieved strong results, with final validation accuracy reaching 95%. The graphs suggest overfitting started after the second epoch.

Model is hosted here: [gnlp_hw1_reranker](https://huggingface.co/greatakela/gnlp_hw1_reranker)

## Training Results Summary

These results indicate high training and validation performance. A consistent 95% validation accuracy suggests two possibilities:
1. **Overfitting** – the model fits the training data too closely, reducing generalization. However, stable high accuracy on validation may indicate that the data represents the target pattern well.
2. **Data Quality** – the dataset may lack diversity or be too small, allowing the model to easily fit it. Increasing data variety and size is important.

I tend to believe the second explanation, as the final dataset was relatively small.

## Repository Structure

```bash
│   README.md - HW1 report
│   requirements.txt
│   __init__.py
│   retrieval_bot.py - main inference logic
│   utilities.py - helper functions
│   app.py - Flask UI app launcher
│
├───Notebooks - data prep and training notebooks
├───templates - HTML template for UI
│       chat.html
├───static - styling for UI
│       style.css
├───data
│       spock_dujour.pkl - fallback lines for low similarity
│       spock_lines_vectorized.pkl - vector DB of context-question pairs
│       spock_lines.pkl - raw data
│       spock_lines_reranker.pkl - reranker dataset
```

## Web Chat Service

The chatbot is built with Flask and launched using `app.py`, which sets up the interface, loads files and models, and initializes the chatbot.

To run locally:  
1. Clone the repo: `https://github.com/greatakela/ChatBot.git`  
2. Create a virtual environment  
3. Install dependencies: `pip install -r requirements.txt`  
4. Launch the app: `python app.py`  
5. The chatbot runs at `http://127.0.0.1:5000`

## Chatbot Evaluation

The bot should be evaluated based on how relevant its replies are to dialogue context. Thus, human evaluation is key.

I tested different encoders:
- [`sentence-transformers/all-mpnet-base-v2`](https://huggingface.co/sentence-transformers/all-mpnet-base-v2)
- [`sentence-transformers/LaBSE`](https://huggingface.co/sentence-transformers/LaBSE)
- [`greatakela/gnlp_hw1_encoder`](https://huggingface.co/greatakela/gnlp_hw1_encoder) — trained specifically on Spock lines

I tested with some sample user inputs to compare retrieval quality:

| **Incoming** | **Greetings, Mr. Spock.** | **What is the logical course of action?** | **Explain your reasoning.** | **What do you think of Captain Kirk?** |
| :---: | :---: | :---: | :---: | :---: |
| **all-mpnet-base-v2** | Live long and prosper. | Logic is the beginning of wisdom, not the end. | Once you have eliminated the impossible, whatever remains, however improbable, must be the truth. | Captain, you almost make me believe in luck. |
| **LaBSE** | Greetings. How may I assist in your endeavors? | It would be illogical to assume that all conditions remain stable. | The universe is vast and full of wonders. It is logical to explore them. | Without followers, evil cannot spread. |
| **gnlp_hw1_encoder** | I assume this greeting is a social convention rather than a necessity? | The needs of the many outweigh the needs of the few. | Superior ability breeds superior ambition. | I fail to comprehend your indignation, sir. I have simply made the logical deduction that you are a liar. |

Interestingly, embeddings from the custom-trained Spock model produced better similarity results than general-purpose models.

It's hard to judge a “winner” from the examples above. I retained the trained model and added intent filtering and a minimum similarity threshold before reranking, to add determinism.

## Conclusion

The analysis shows that the developed chatbot model is effective for this task. However, further testing on larger and more diverse data is necessary to better understand its generalization capabilities and limits.

## Web Service Deployment

The service was containerized using Docker on a local machine. I deployed it to a virtual server on **Kamatera**, running the uploaded Docker container. The chatbot is accessible at:  
👉 **http://185.53.209.56:5000/**

The Docker image ended up being 13 GB even without GPU-related packages. There's room for optimization.  
**VM Specs**: 2 CPUs, 2 GB RAM, 80 GB storage

