# recommendations_system.py

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix
import re

# Load Course Features
course_features = pd.read_csv("Course_Features.csv")

def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', str(text))
    text = text.lower().strip()
    return text

course_features['Processed Description'] = course_features['Course Description'].apply(preprocess_text)
course_features['Normalized Name'] = course_features['Course Name'].apply(preprocess_text)

# TF-IDF Matrix for Content-Based Filtering
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(course_features['Processed Description'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Load Student Course Data
student_course_data = pd.read_csv("Sanitized_Synthetic_Student_Course_Data.csv")
student_course_data['all_courses'] = student_course_data[[f'sem_{i}' for i in range(1, 7)]].apply(
    lambda row: ', '.join(row.dropna().astype(str)), axis=1
)
student_course_data['all_courses'] = student_course_data['all_courses'].str.split(', ').apply(lambda courses: list(set(courses)))

# Create User-Course Matrix
all_unique_courses = set(course for courses in student_course_data['all_courses'] for course in courses)
user_course_matrix = pd.DataFrame(0, index=student_course_data['user_id'], columns=sorted(all_unique_courses))

for idx, courses in zip(student_course_data['user_id'], student_course_data['all_courses']):
    user_course_matrix.loc[idx, courses] = 1

sparse_user_course_matrix = csr_matrix(user_course_matrix.values)
user_similarity_matrix = cosine_similarity(sparse_user_course_matrix)

# --- HYBRID FUNCTION ---
def hybrid_recommendation_for_new_user(stream, semester, courses_taken_list, top_n=5):
    global course_features, tfidf_matrix, cosine_sim, student_course_data, user_course_matrix, user_similarity_matrix

    # Normalize input
    normalized_input = [preprocess_text(course) for course in courses_taken_list]
    normalized_course_names = course_features['Normalized Name'].tolist()
    all_courses = list(user_course_matrix.columns)

    # Build new user vector
    new_user_vector = pd.Series(0, index=all_courses)
    matched_courses = []

    for course in courses_taken_list:
        course_clean = preprocess_text(course)
        match_idx = next((i for i, name in enumerate(normalized_course_names) if name == course_clean), None)
        if match_idx is not None:
            matched_name = course_features.iloc[match_idx]['Course Name']
            matched_courses.append(matched_name)
            if matched_name in new_user_vector:
                new_user_vector[matched_name] = 1

    if not matched_courses:
        return "No valid courses found for your input."

    # Extend user-course matrix
    extended_matrix = pd.concat([user_course_matrix, pd.DataFrame([new_user_vector])], ignore_index=True)
    sparse_extended = csr_matrix(extended_matrix.values)
    sim_matrix = cosine_similarity(sparse_extended)

    new_user_index = len(extended_matrix) - 1
    user_sim_scores = sim_matrix[new_user_index]
    user_course_pref = np.dot(user_sim_scores, extended_matrix.values)
    user_course_pref /= user_sim_scores.sum() if user_sim_scores.sum() != 0 else 1

    course_indices = [course_features[course_features['Course Name'] == name].index[0] for name in matched_courses]
    content_scores = cosine_sim[course_indices].mean(axis=0)

    aligned_content_scores = pd.Series(content_scores, index=course_features['Course Name'])
    aligned_content_scores = aligned_content_scores.reindex(all_courses, fill_value=0).values

    hybrid_scores = user_course_pref + aligned_content_scores
    recommendations = pd.Series(hybrid_scores, index=all_courses)

    recommendations = recommendations[~recommendations.index.isin(matched_courses)]
    recommendations = recommendations.sort_values(ascending=False).head(top_n)

    return recommendations.index.tolist(), recommendations.values.tolist()
