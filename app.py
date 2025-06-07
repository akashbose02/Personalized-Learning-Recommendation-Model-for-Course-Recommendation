# app.py
from flask import Flask, render_template, request
import pandas as pd
from recommendations_system import hybrid_recommendation_for_new_user, course_features

app = Flask(__name__)
app.jinja_env.globals.update(zip=zip)

@app.route("/", methods=["GET", "POST"])
def index():
    recommendations, scores, error = [], [], None

    # Load all course names from the CSV (sorted for cleaner UI)
    all_courses = sorted(course_features['Course Name'].dropna().unique().tolist())
    selected_courses = []

    if request.method == "POST":
        stream = request.form.get("stream", "").strip().upper()
        semester = request.form.get("semester", "").strip()
        selected_courses = request.form.getlist("courses_taken")  # Multi-select

        if not stream or not semester or not selected_courses:
            error = "All fields are required."
        else:
            result = hybrid_recommendation_for_new_user(stream, semester, selected_courses)

            if isinstance(result, str):
                error = result
            else:
                recommendations, scores = result

    return render_template("index.html",
                           recommendations=recommendations,
                           scores=scores,
                           error=error,
                           all_courses=all_courses,
                           selected_courses=selected_courses)

if __name__ == "__main__":
    app.run(debug=True)
