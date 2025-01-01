import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Loading the dataset
mmh = pd.read_csv('mxmh_survey_results.csv')

# Data cleaning and preparation
# Filter relevant columns
mmh = mmh[['Age', 'Primary streaming service', 'Hours per day', 'Fav genre', 'Depression']]

# Drop rows with missing values in relevant columns
mmh['Depression'] = pd.to_numeric(mmh['Depression'], errors='coerce')

# Analyse Depression by Genre
genre_depression = mmh.groupby('Fav genre')['Depression'].mean().sort_values()

# Plot bar graph
plt.figure(figsize=(12,6))
genre_depression.plot(kind='bar', color='skyblue')
plt.title('Average Depression Scores by Music Genre', fontsize=16)
plt.ylabel('Average Depression in Score', fontsize=14)
plt.xlabel('Music Genre', fontsize=14)
plt.xticks(rotation=45, fontsize=12)
plt.show()


# Grouping ages into ranges
age_groups = [0, 18, 25, 35, 50, 100]
age_labels = ['0-18', '19-25', '26-35', '36-50', '50+']
mmh['Age Range'] = pd.cut(mmh['Age'], bins=age_groups, labels=age_labels, right=False)

# Grouping by Age Range and Favorite Genre
age_genre_depression = mmh.groupby(['Age Range', 'Fav genre'])['Depression'].mean().unstack()

# Analyse by Age Range and Genre
plt.figure(figsize=(14, 8))
sns.heatmap(age_genre_depression, annot=True, fmt=".1f", cmap="coolwarm")
plt.title('Depression Score by Age Range and Genre', fontsize=16)
plt.ylabel('Age Range', fontsize=14)
plt.xlabel('Music Genre', fontsize=14)
plt.show()

# Analyse by streaming service
streaming_genre_depression = mmh.groupby(['Primary streaming service', 'Fav genre']) ['Depression'].mean().unstack()

# Heatmap: Depression by Streaming Service and Genre
plt.figure(figsize=(14,8))
sns.heatmap(streaming_genre_depression, annot=True, fmt=".1f", cmap="viridis")
plt.title('Depression Scores by Streaming Service and Genre', fontsize=16)
plt.ylabel('Streaming Service', fontsize=14)
plt.xlabel('Music Genre', fontsize=14)
plt.show()

# Analyze by Hours Listening per Day
bins_hours = [0, 1, 3, 5, 10, 24]
labels_hours = ['<1', '1-3', '3-5', '5-10', '10+']
mmh['Listening Hours Range'] = pd.cut(mmh['Hours per day'], bins=bins_hours, labels=labels_hours, right=False)
hours_genre_depression = mmh.groupby(['Listening Hours Range', 'Fav genre'])['Depression'].mean().unstack()

# Heatmap: Depression by Listening Hours and Genre
plt.figure(figsize=(14, 8))
sns.heatmap(hours_genre_depression, annot=True, fmt=".1f", cmap="magma")
plt.title('Depression Scores by Listening Hours and Genre', fontsize=16)
plt.ylabel('Listening Hours Range', fontsize=14)
plt.xlabel('Music Genre', fontsize=14)
plt.show()

# Summary: Least and Most Depressed Genres
most_depressed_genres = genre_depression.nlargest(5)
least_depressed_genres = genre_depression.nsmallest(5)

print("Top 5 Genres with the Most Depressed People:")
print(most_depressed_genres)

print("\nTop 5 Genres with the Least Depressed People:")
print(least_depressed_genres)

