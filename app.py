
from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)

# =========================================================
# LOAD TRAINED MODEL
# =========================================================

with open("placement_model.pkl", "rb") as file:
    model = pickle.load(file)


# =========================================================
# HOME PAGE
# =========================================================

@app.route("/")
def home():
    return render_template("index.html")


# =========================================================
# PREDICTION
# =========================================================

@app.route("/predict", methods=["POST"])
def predict():

    try:

        # -------------------------------------------------
        # GET DATA FROM FORM
        # -------------------------------------------------

        data = {
            "Gender": request.form["Gender"],
            "City": request.form["City"],
            "CollegeTier": request.form["CollegeTier"],
            "Stream": request.form["Stream"],
            "Specialisation": request.form["Specialisation"],
            "Hostel": request.form["Hostel"],
            "HistoryOfBacklogs": request.form["HistoryOfBacklogs"],

            "SGPA_Sem1": float(request.form["SGPA_Sem1"]),
            "SGPA_Sem2": float(request.form["SGPA_Sem2"]),
            "SGPA_Sem3": float(request.form["SGPA_Sem3"]),
            "SGPA_Sem4": float(request.form["SGPA_Sem4"]),
            "SGPA_Sem5": float(request.form["SGPA_Sem5"]),
            "SGPA_Sem6": float(request.form["SGPA_Sem6"]),
            "SGPA_Sem7": float(request.form["SGPA_Sem7"]),
            "SGPA_Sem8": float(request.form["SGPA_Sem8"]),

            "CGPA": float(request.form["CGPA"]),
            "AttendancePercent": float(request.form["AttendancePercent"]),

            "Internships": int(request.form["Internships"]),
            "Projects": int(request.form["Projects"]),
            "Workshops": int(request.form["Workshops"]),
            "Certifications": int(request.form["Certifications"]),
            "Publications": int(request.form["Publications"]),

            "AptitudeTestScore": float(
                request.form["AptitudeTestScore"]
            ),

            "SoftSkillsRating": float(
                request.form["SoftSkillsRating"]
            ),

            "CodingTestScore": float(
                request.form["CodingTestScore"]
            ),

            "MockInterviewScore": float(
                request.form["MockInterviewScore"]
            ),

            "ExtraCurricular": int(
                request.form["ExtraCurricular"]
            ),

            "CGPA_Tier": request.form["CGPA_Tier"]
        }

        # -------------------------------------------------
        # CREATE DATAFRAME
        # -------------------------------------------------

        input_data = pd.DataFrame([data])

        # -------------------------------------------------
        # MACHINE LEARNING PREDICTION
        # -------------------------------------------------

        prediction = model.predict(input_data)[0]

        probabilities = model.predict_proba(input_data)[0]

        # 0 = Not Placed
        # 1 = Placed

        probability = probabilities[1] * 100

        # -------------------------------------------------
        # RESULT
        # -------------------------------------------------

        if prediction == 1:
            result = "Likely to be Placed"
        else:
            result = "Needs Improvement"

        # -------------------------------------------------
        # RECOMMENDATIONS
        # -------------------------------------------------

        recommendations = []

        if data["CGPA"] < 7:
            recommendations.append(
                "Improve your CGPA"
            )

        if data["AttendancePercent"] < 75:
            recommendations.append(
                "Improve your attendance"
            )

        if data["Internships"] == 0:
            recommendations.append(
                "Try to complete at least one internship"
            )

        if data["Projects"] < 2:
            recommendations.append(
                "Build more projects"
            )

        if data["CodingTestScore"] < 60:
            recommendations.append(
                "Practice coding regularly"
            )

        if data["AptitudeTestScore"] < 60:
            recommendations.append(
                "Improve aptitude preparation"
            )

        if data["MockInterviewScore"] < 60:
            recommendations.append(
                "Practice mock interviews"
            )

        if data["SoftSkillsRating"] < 3:
            recommendations.append(
                "Improve communication and soft skills"
            )

        if data["Certifications"] == 0:
            recommendations.append(
                "Consider completing relevant certifications"
            )

        # -------------------------------------------------
        # SEND RESULT TO HTML
        # -------------------------------------------------

        return render_template(
            "index.html",
            prediction=result,
            probability=round(probability, 2),
            recommendations=recommendations
        )

    except Exception as e:

        print("ERROR:", e)

        return render_template(
            "index.html",
            error=str(e)
        )


# =========================================================
# RUN FLASK
# =========================================================

if __name__ == "__main__":
    app.run(debug=True)

