import pandas as pd



def loadAndMergeMovieData():
    '''
    This function loads the movie name dataset and the user rating dataset
    Returns merged dataframe with user-movie ratings
    '''
    rating_cols = ['user_id', 'movie_id', 'rating']
    ratings = pd.read_csv('Z:/ML/DataScience/DataScience/ml-100k/u.data', sep = '\t', 
                            names = rating_cols, usecols = range(3))

    movie_cols = ['movie_id', 'title']
    movies = pd.read_csv('Z:/ML/DataScience/DataScience/ml-100k/u.item', sep = '|', 
                        names = movie_cols, usecols = range(2))

    ratings = pd.merge(movies, ratings)
    return ratings

def createRatingsPivot(ratings):
    '''
    This function creates a pivot table with user_id as index and the rating 
    for each movie given by the user as column values
    '''
    userRatings = ratings.pivot_table(index = ['user_id'], columns = ['title'], values = 'rating')
    return userRatings
    
def createCorrMatrix(userRatings):
    '''
    This function creates the correlation matrix which gives the similarity of ratings
    for each pair of movies rated by a user
    This uses the Pearson correlation scores and ignores the movie data that are 
    rated less than 100 people
    '''
    corrMatrix = userRatings.corr(method='pearson', min_periods=100)
    return corrMatrix

def selectUserForRecommendation(userRatings):
    '''
    This function selects the movie rating data for the user to whom
    the recommendations are to be made
    '''
    userDf = pd.read_csv('Z:/ML/DataScience/DataScience/ml-100k/userid.csv')
    userIndex = userDf['userId']
    selectedUser = userRatings.loc[userIndex].dropna()
    return selectedUser

def getSimilarMovies(userData,corrMatrix):
    '''
    This function creates a dataframe of similar movies based on the 
    user's ratings
    '''
    simCandidates = pd.Series()
    for i in range(0, len(userData.index)):
        
    # Retrieving similar movies to that of the user rated
        similarMovies = corrMatrix[userData.index[i]].dropna()
    # Scaling the similarity
        similarMovies = similarMovies.map(lambda x: x * userData[i])
    # Add the score to the list of similarity candidates
        similarMovieCandidates = simCandidates.append(similarMovies)
    
    similarMovieCandidates.sort_values(inplace = True, ascending = False)
    #Grouping the repeated results and similarity scores are added
    similarMovieCandidates = similarMovieCandidates.groupby(similarMovieCandidates.index).sum()
    similarMovieCandidates.sort_values(inplace = True, ascending = False)
    #Removing the movies data that are rated by the user
    filteredSimilarMovies = similarMovieCandidates.drop(userData.index)
    return filteredSimilarMovies


def main():
    ratings = loadAndMergeMovieData()
    userRatings = createRatingsPivot(ratings)
    corrMatrix = createCorrMatrix(userRatings)
    userData = selectUserForRecommendation(userRatings)
    similarMovies = getSimilarMovies(userData,corrMatrix)
    similarMovies.to_csv('Z:/ML/DataScience/DataScience/ml-100k/similarMovies.csv', sep  =',')
    
    
if __name__ == 'main':
    main()
