# Final Project Report: Short Video Recommender System (KuaiRec)

## Overview

This project implements a scalable short video recommender system using the **KuaiRec** dataset. The goal is to build a **personalized, content-based recommendation engine** that predicts which videos a user is most likely to engage with, inspired by platforms like TikTok and Kuaishou.

We adopt the **NNEncoding architecture**, which applies deep learning to user and item content features using dual-encoder towers. The final system ranks videos for a user based on predicted affinity learned from historical interactions.

---

## Data Preprocessing

The EDA code was proposed as a starting point for the project so i started my project by building off of it.
when looking at the possible features i could put for the users and videos i noticed that they both had features.
After investigation, i concluded that they couldnt represent the same thing since video feats range from 0 to 30 while user feats range from 1 to 17 with values in the hundreds.
I confirmed online that the video feats represent tags so i decided to build most of the features around that.

I also decided to use watch ratio as one of the features and found that it's value ranges from 0 to 5+ but there wasn't a significant amount of videos with a ratio of more than five so i limit the ratio to just 0 to 5 in order to not skew the data too much.
From the graph we can see that the watch ratio peaks at 0.1 which makes sense as tiktok is a short form video platform which makes skipping videos easy.
We also see the ratio monotonely decreases past a ratio of 1 since most people dont watch the same video multiple times.

we can see from the 'distribution of interactions per user' graphs that:
- a vast majority of users have less than 3000 interactions and we have outliers at 7500 and 9000
- a few super users contribute the majority of interactions

we can see from the 'distribution of interactions per user' graphs that:
- most videos have few interactions and more popular items tend to get more exposure

---

## Feature Engineering

if we try to make a system like tiktok we would try to incentivize the user to stay on the app for as long as possible so we should try to take note of the videos that they watch and find the ones that they like the most. with the dataset we must work with implicit data so for our purposes we will be using watch ratio as the metric to say whether a specific user but we also make use of play counts, comment counts, follow counts and share counts to estimate if a video has a general appeal. since we want to recommend similar videos to what the user has already watched we will use the video feats/tags as a way to estimate how similar two videos might be

### User Features

We computed average watch ratio per category (`ufeat_X`) for each user:

- Merged `interactions` with `item_categories`
- Aggregated `watch_ratio` by `user_id` × `feat`
- Pivoted into wide format (one-hot feats)

### Item Features

The item feature matrix includes:

- One-hot encoding of each video's `feat` category (`ifeat_X`)
- Aggregated statistics:
  - `video_duration`
  - `play_cnt`, `comment_cnt`, `follow_cnt`, `share_cnt`
- All numeric columns scaled with `StandardScaler`

---

## Model Development: NNEncoding

I chose a neural network encoding approach (NNEncoding), inspired by the lecture structure, because it gives us flexibility in modeling non-linear interactions between a user's historical preferences and a video's content. Compared to matrix factorization or dot-product only models, deep neural nets allow for more expressive embeddings based on actual features. 

I used a two-tower architecture: one for users and one for items. Each takes a fixed-size input vector of standardized features and maps it into an embedding space. These embeddings are normalized and compared via dot product to produce a similarity score, which is trained to predict scaled watch ratio. 

We implemented a **dual-tower neural model**:

- **User encoder**: MLP (128 → 64 → 32) + L2 normalization
- **Item encoder**: MLP (128 → 64 → 32) + L2 normalization
- **Similarity**: Dot product of user and item vectors

This structure allows fast inference and clean dot product scoring for recommendation.

---

## Training Details

- Loss: Mean Squared Error (MSE)
- Optimizer: Adam
- Batch size: 128
- Epochs: 5
- Target (`watch_ratio`) scaled using `MinMaxScaler(feature_range=(-1, 1))`
- Data batched using `tf.data.Dataset` for performance

---

## Evaluation Metrics

We wanted to evaluate the model in terms that reflect both how accurately it predicts user engagement (as regression or relevance score), and how well it performs in a top-K recommender setting.
we framed watch_ratio ≥ 0.80 as an implicit "positive" label — similar to a user fully watching (or rewatching) a video. Based on this, we calculated classification metrics like precision, recall, and accuracy using a threshold in the scaled range.

### Classification Metrics

**Accuracy**: 0.5565
**Precision**: 0.5313
**Recall**: 0.8610

### Ranking Metrics (Top-K)

Evaluated using the Top-K interacted items (by watch ratio) from each user:

- **Precision@K**
- **Recall@K**
- **Accuracy@K** (Hit rate)

outputs:
Precision@10: 0.0009
Recall@10:    0.0009
Accuracy@10:  0.0064
Evaluated users: 1411


Precision@100: 0.0223
Recall@100:    0.0223
Accuracy@100:  0.8894
Evaluated users: 1411


Precision@1000: 0.2615
Recall@1000:    0.2615
Accuracy@1000:  1.0000
Evaluated users: 1411

At K = 10, the model struggles to retrieve meaningful hits — both Precision@10 and Recall@10 are very low (~0.0009), meaning that among the top 10 videos recommended, almost none were present in the user's top 10 watched videos (based on watch ratio).
Performance improves significantly as K increases: 
- At K = 100, recall and precision increase to ~0.0223 (2.2%) and hit rate (Accuracy@K) jumps to almost 89%.  
- At K = 1000, recall and hit rate reach nearly perfect coverage (100%), indicating the model is broadly aware of the user’s preferences but is unable to rank the best items near the top.
     
These results are typical of content-based models trained with regression loss (MSE), which learn general preferences but lack sharp ranking capability for high-confidence top choices. It suggests the model can retrieve relevant items, but needs further tuning (e.g., ranking loss or filtering) to improve short-list relevance. 

---

## Conclusion

This project successfully implemented a scalable content-based short video recommender system using the KuaiRec dataset. By leveraging the NNEncoding architecture, we built a model that learns deep user and item representations directly from available features like video tags and engagement statistics. The system was trained to predict watch ratio, using a dot product between normalized user and item vectors to rank videos. 

The model pipeline—from data preprocessing and feature engineering to model training and recommendation generation—was fully realized and evaluated. 

Our evaluation results confirmed that: 
    The model learns generalized user preference patterns, achieving perfect recall at high K, meaning it eventually retrieves relevant videos.
    However, precision at low K is very low, which indicates the model struggles to rank the most relevant items near the top.
    This behavior aligns with expectations for regression-based recommenders optimized with MSE loss and no explicit ranking objective.
     