from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix


# Load the user data
user_data_path = '/content/drive/MyDrive/4th Year Project/Sanitized_Synthetic_Student_Course_Data.csv'
user_data = pd.read_csv(user_data_path)
#Cell 9

# Step 1: Combine courses from Semester 3 & 4 and Semester 5 & 6 into a single list for each user
user_data['all_courses'] = user_data[['sem_3', 'sem_4', 'sem_5', 'sem_6']].apply(
    lambda row: ', '.join(row.dropna().astype(str)), axis=1
)
user_data['all_courses'] = user_data['all_courses'].str.split(', ').apply(lambda courses: list(set(courses)))

# Create a set of all unique courses
all_unique_courses = set(course for courses in user_data['all_courses'] for course in courses)

# Construct the binary user-course matrix
user_course_matrix = pd.DataFrame(0, index=user_data['user_id'], columns=sorted(all_unique_courses))
for idx, courses in zip(user_data['user_id'], user_data['all_courses']):
    user_course_matrix.loc[idx, courses] = 1

#Cell 10
# Step 2: Calculate user-user similarity matrix using cosine similarity
sparse_user_course_matrix = csr_matrix(user_course_matrix.values)
user_similarity_matrix = cosine_similarity(sparse_user_course_matrix)


# Create a DataFrame for the user-user similarity matrix
user_similarity_df = pd.DataFrame(
    user_similarity_matrix,  # The matrix computed using cosine similarity
    index=user_course_matrix.index,  # Use user IDs as the row index
    columns=user_course_matrix.index  # Use user IDs as the column index
)
# Display the first few rows of the similarity matrix
print(user_similarity_df.head(10))
#Cell 11
# Step 3: Predict course preferences
predicted_preferences = user_similarity_matrix @ user_course_matrix.values
similarity_sum = user_similarity_matrix.sum(axis=1).reshape(-1, 1)
normalized_predictions = predicted_preferences / similarity_sum


# Step 4: Take user input and generate recommendations
try:
    user_id = int(input("Enter the User ID for recommendations: "))

    # Ensure the user ID is valid
    if user_id in user_data['user_id'].values:
        # Get predicted preferences for the user
        user_recommendations = pd.Series(normalized_predictions[user_id - 1], index=user_course_matrix.columns)
        user_recommendations = user_recommendations.sort_values(ascending=False)

        # Filter out courses the user has already taken
        already_taken = set(user_data.loc[user_data['user_id'] == user_id, 'all_courses'].values[0])
        new_recommendations = [course for course in user_recommendations.index if course not in already_taken]

        # Display the top 5 recommendations
        print(f"Top 5 Recommendations for User ID {user_id}: \n {new_recommendations[:5]}")
    else:
        print(f"User ID {user_id} not found in the dataset.")
except ValueError:
    print("Invalid User ID. Please enter a valid numeric User ID.")
    # Cell 12

    # Create a similarity DataFrame
    user_similarity_df = pd.DataFrame(
        user_similarity_matrix,
        index=user_data['user_id'],
        columns=user_data['user_id']
    )

    # Display the most similar users for each user
    most_similar_users = user_similarity_df.apply(
        lambda row: row.nlargest(2).index.tolist()[1], axis=1
    )  # Get the second-largest similarity score (ignoring the user themselves)

    # Show the most similar user for each user
    similarity_pairs = pd.DataFrame({
        "User": user_similarity_df.index,
        "Most Similar User": most_similar_users
    })

    # Display the similarity pairs directly
    print(similarity_pairs.head())  # Show the first few rows
    # Cell 13

    # content based filtering recommender system

    from sklearn.feature_extraction.text import TfidfVectorizer

    # Load the dataset
    file_path = '/content/drive/MyDrive/4th Year Project/Course_Features.csv'
    course_features = pd.read_csv(file_path)

    course_features.head()
    # Cell 14
    # Function to preprocess text
    import re


    def preprocess_text(text):
        text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove special characters
        text = text.lower()  # Convert to lowercase
        return text


    # Preprocess course descriptions
    course_features['Processed Description'] = course_features['Course Description'].apply(preprocess_text)

    course_features['Processed Description'].head()
    # Cell 15

    # Compute the TF-IDF matrix for the course descriptions
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform(course_features['Processed Description'])

    # Compute cosine similarity between courses
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)


    # Cell 15

    # Function to get similar courses
    def get_similar_courses(course_name, cosine_sim=cosine_sim, df=course_features):
        # Find the index of the course
        idx = df[df['Course Name'] == course_name].index[0]

        # Get similarity scores for all courses with the selected course
        sim_scores = list(enumerate(cosine_sim[idx]))

        # Sort courses by similarity scores in descending order
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

        # Get the top 5 similar courses (excluding itself)
        sim_scores = sim_scores[1:6]

        # Get the course names and their similarity scores
        similar_courses = [(df.iloc[i[0]]['Course Name'], i[1]) for i in sim_scores]

        return similar_courses


    # Cell 16

    # Test with a course name
    example_course = "Introduction to Machine Learning"
    similar_courses = get_similar_courses(example_course)

    # Print the similar courses
    for course, score in similar_courses:
        # Cell 1
        print(f"Course: {course}, Similarity Score: {score:.2f}")
        # Cell 17
        # Hybrid recommender system

        # Load the datasets
        course_features_path = '/content/drive/MyDrive/4th Year Project/Course_Features.csv'
        student_course_data_path = '/content/drive/MyDrive/4th Year Project/Sanitized_Synthetic_Student_Course_Data.csv'

        course_features = pd.read_csv(course_features_path)
        student_course_data = pd.read_csv(student_course_data_path)


        # Preprocess text for content-based filtering
        def preprocess_text(text):
            text = re.sub(r'[^a-zA-Z\s]', '', text)
            text = text.lower()
            return text


        course_features['Processed Description'] = course_features['Course Description'].apply(preprocess_text)
        # Cell 18
        # Content-Based Filtering: Compute TF-IDF and cosine similarity
        tfidf_vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf_vectorizer.fit_transform(course_features['Processed Description'])
        cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

        # Collaborative Filtering: User-course matrix and similarity
        student_course_data['all_courses'] = student_course_data[['sem_3', 'sem_4', 'sem_5', 'sem_6']].apply(
            lambda row: ', '.join(row.dropna().astype(str)), axis=1
        )
        student_course_data['all_courses'] = student_course_data['all_courses'].str.split(', ').apply(
            lambda courses: list(set(courses)))

        all_unique_courses = set(course for courses in student_course_data['all_courses'] for course in courses)
        user_course_matrix = pd.DataFrame(0, index=student_course_data['user_id'], columns=sorted(all_unique_courses))

        for idx, courses in zip(student_course_data['user_id'], student_course_data['all_courses']):
            user_course_matrix.loc[idx, courses] = 1

        sparse_user_course_matrix = csr_matrix(user_course_matrix.values)
        user_similarity_matrix = cosine_similarity(sparse_user_course_matrix)


        # Cell 19

        def hybrid_recommendation_with_scores(user_id, top_n=5):
            if user_id not in student_course_data['user_id'].values:
                return f"User ID {user_id} not found."

            # Collaborative Filtering Component
            user_index = student_course_data[student_course_data['user_id'] == user_id].index[0]
            user_sim_scores = user_similarity_matrix[user_index]
            user_course_pref = np.dot(user_sim_scores, user_course_matrix.values)
            user_course_pref /= user_sim_scores.sum() if user_sim_scores.sum() != 0 else 1

            # Content-Based Filtering Component
            user_courses = student_course_data.loc[user_index, 'all_courses']
            course_indices = [course_features[course_features['Course Name'] == course].index[0]
                              for course in user_courses if course in course_features['Course Name'].values]

            if not course_indices:
                return f"No valid courses found for User ID {user_id}."

            content_scores = cosine_sim[course_indices].mean(axis=0)

            # Align Content-Based Scores to User-Course Matrix
            aligned_content_scores = pd.Series(content_scores, index=course_features['Course Name'])
            aligned_content_scores = aligned_content_scores.reindex(user_course_matrix.columns, fill_value=0).values

            # Combine Scores
            hybrid_scores = user_course_pref + aligned_content_scores

            # Get Course Recommendations and Similarity Scores
            recommendations = pd.Series(hybrid_scores, index=user_course_matrix.columns)
            already_taken = set(user_courses)
            recommendations = recommendations[~recommendations.index.isin(already_taken)]
            recommendations = recommendations.sort_values(ascending=False).head(top_n)

            # Return recommendations and similarity scores
            return recommendations.index.tolist(), recommendations.values.tolist()


        # Cell 20
        try:
            # Prompt the user to enter their user ID
            user_id = int(input("Enter your User ID: "))

            # Get recommendations with similarity scores
            recommended_courses, similarity_scores = hybrid_recommendation_with_scores(user_id, top_n=5)

            # Print recommendations and similarity scores
            print(f"\nTop 5 Recommended Courses for User ID {user_id}:")
            for course, score in zip(recommended_courses, similarity_scores):
                print(f"Course: {course}, Similarity Score: {score:.4f}")

        except ValueError:
            print("Invalid input. Please enter a numeric User ID.")
        except Exception as e:
            print(f"An error occurred: {e}")
