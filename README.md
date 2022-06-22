# Game-Recommender

# Recommendation System with *Content-based Filtering*

## Background
Game is a tool used to play, an item or something that is usually used for entertainment or pleasure, and sometimes used as an educational tool. Play is distinct from work, which is usually done for a wage, and from art, which is more often an expression of aesthetic or ideological elements. However, the distinction is not clear, and many games are also considered works (such as professional sports players or spectator games) or art (such as puzzles or games involving an artistic layout such as Mahjong, solitaire, or some video games).

Games are very relevant to the current condition, namely a pandemic where most people use games as a means to relieve boredom at home. There are many genres of games such as action, sports, music, etc. Games also have certain platforms such as those played on PC, PS, Mobile, etc. Usually people with certain game genres, such as action, tend to play games with the action genre as well. Likewise with platforms, people who usually play games on certain platforms such as PCs, tend to play games that are on the same platform.

In this machine learning project, a recommendation system model will be made to predict preferred games based on other games that have similarities or by using *content-based filtering* techniques with several variables such as platform, year of release, genre, etc.

People who play certain game genres on certain platforms tend to be stuck with the same genres and platforms. Or because they are satisfied with games launched by certain publishers, they will look forward to games that are also launched by that publisher. Therefore, here a game recommendation system will be created that can predict the games that users might like based on genre, platform, publisher, and other variables that have similarities or with *content-based filtering*.

## Business Understanding 
To create a recommendation system that can recommend games that users might like, a content-based filtering recommendation system model technique will be used with the KNearestNeighbors and CosineSimilarity algorithms.

- Problems to solve:
Get recommendations for specific games that users might like
- The purpose of making Machine Learning Model:
The model can recommend certain games with a high degree of similarity (>90%).
- Solution:
Creating Machine Learning with the KNearestNeighbors Algorithm that can recommend games based on the k-similarity of certain game features.

## Data Understanding 
The dataset used to recommend games is a dataset from Kaggle. This dataset has been used to predict stroke with 242 different model algorithms. This dataset has 16719 samples or rows with 16 features or columns to create a game recommendation system with *content-based filtering*. This dataset was created by [Rush Kirubi](https://www.kaggle.com/rush4ratio).  

Dataset Link: [Game Sales with Ratings Dataset](https://www.kaggle.com/rush4ratio/video-game-sales-with-ratings)  

Dataset features: 
1. Name : The name of the game to be recommended [str]
2. Platform : The platform on which the game is provided (there are 31 different platforms) [str]
3. Year_of_Release : Game release year [str]
4. Genre: Genre or type of game that contains 12 different genres [str]
5. Publisher : Publisher who publishes game [str]
6. NA_Sales : Game sales in the North America region in million [float]
7. EU_Sales : Game sales in the Europe region in million [float]
8. JP_Sales : Sales of games in the Japan region in million [float]
9. Other_Sales : Game sales in regions other than North America, Europe and Japan in million [float]
10. Global_Sales : Total game sales in all regions in million [float]
11. Critic_Score : Aggregate score given by metacritic reviewer [float]
12. Critic_Count : Total metacritic reviewers who rate the game [float]
13. User_Score : Aggregate value given by user [float]
14. User_Count : Total users who rate the game [float]
15. Developer : Companies that collaborate in making games [str]
16. Rating : ESRB Rating provides a rating that usually describes a certain group within a certain age range ('E', 'M', 'T', 'E10+', 'K-A', 'AO', 'EC', 'RP ') [str]

## Data Preparation 
For the data preparation stage, several steps have been carried out, namely by removing empty data in each column, eliminating unnecessary columns such as the Critic_Score, Critic_Count, and User_Count columns and the Global_Sales column, removing unique elements in columns with unique elements of little value such as the Platform column, remove duplicate data, perform one-hot encoding on categorical data, and standardize numeric columns with MinMaxScaler.

Steps are needed, such as removing empty data so that the calculation or algorithm does not get an error. For the stages of removing certain columns such as the Critic_Score, Critic_Count, and User_Count columns because there are too many empty data and the Global_Sales column because it has been described by sales in the divided region, it is necessary so that data that can damage the model can be discarded. After that, remove the unique element in the column with a unique element that has little value such as the Platform column with an amount below 350 so that data quality can improve. Then remove the duplicate data so that the model quality is also good. The standardization stage is carried out so that the features are not slamming in value with other features using MinMaxScaler and one-hot encoding to make categorical features numerical.

## Modeling 
Machine learning modeling to recommend certain games to users is using the K-NearestNeighbors algorithm. This algorithm works based on existing features and similarities between these features to assess whether a particular game can be recommended when a user plays a game.

With the existing data and after processing the data, the features that affect the game recommendations are taken. Some of the features used are Platform, Year_of_Release, Game, Publisher, NA_Sales, EU_Sales, JP_Sales, Other_Sales, User_Score, and Rating.

At the beginning of making the model, the K-NearestNeighbors model is used with the set of metric parameters used, namely Euclidean distance and a recommendation system function is made for the top 5 games when a game is given. Created a list of recommended game names and their similarities where 100% is subtracted by the euclidean distance of the recommended game. Then the list is entered in the DataFrame so that it can be easily understood by the user.

Using the initial model, predictions are made on the game with the name Final Fantasy IX or loc[111] on the game name DataFrame. The results are in the form of game recommendations such as Final Fantasy VIII, Final Fantasy Tactics, Xenogears, Tales of Destiny II, and Chrono Cross, all of which have similarities based on euclidean distance above 98.5%. A pretty good recommendation for an early model with several similar games is indicated by the name of the recommended game that is the same as the game being played.

Machine learning is developed using a different algorithm, which uses cosine similarity. After making a cosinesimilarity dataframe and making a recommendation function with cosine similarity, the same as the initial model, the results of the game recommendations are entered in a dataframe that contains the name of the recommended game and its similarity using cosinesimilarity.

## Model Evaluation
For the evaluation of the model, two methods or methods of model evaluation were used, namely the Calinski-Harabasz Score and the Davies-Bouldin Score.

Calinski-Harabasz or also known as the Variance Ratio Criterion, is the ratio of the sum of the inter-cluster dispersion and inter-cluster dispersion for all clusters, the higher the score, the better the performance. Obtained a score of 5.09 for the Calinski-Harabasz score which is quite small for the recommendation system model.

The second method is to use the evaluation of the Davies-Bouldin Score. This score indicates the average 'similarity' between clusters, where similarity is a measure that compares the distance between clusters with the size of the cluster itself. A lower Davies-Bouldin score is associated with a model that has better separation between clusters. Obtained a score of 2.93 for the Davies-Bouldin score which is quite high for the recommendation system model. 

It turns out that by using a different algorithm, the results of the Final Fantasy IX game prediction or loc[111] on the DataFrame of the game name are the same as the initial model using KNearestNeighbors. Even though the recommended games such as Final Fantasy VIII, Final Fantasy Tactics, Xenogears, Tales of Destiny II, and Chrono Cross have a cosinesimilarity score above 0.82, it can be concluded that the successful model predicts games that users might like.

### References 
- Scikit-learn Docummentation: [https://scikit-learn.org/stable/modules/classes.html](https://scikit-learn.org/stable/modules/classes.html) 
- Report References: [https://github.com/fahmij8/ML-Exercise/blob/main/MLT-2/MLT_Proyek_Submission_2.ipynb](https://github.com/fahmij8/ML-Exercise/blob/main/MLT-2/MLT_Proyek_Submission_2.ipynb) 
- Wikipedia: [Permainan (Game)](https://id.wikipedia.org/wiki/Permainan#Jenis_permainan) 
- Dataset: [Game Sales with Ratings Dataset](https://www.kaggle.com/rush4ratio/video-game-sales-with-ratings)
