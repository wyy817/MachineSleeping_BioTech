from flask import Blueprint, render_template, request
from flask_login import login_required
from app.llm import mock_llm, further_questions  # Importing the mock LLM functions
from app.models import read_datasets, write_dataset

bp = Blueprint("personal", __name__, url_prefix="/personal")


@bp.route("/consult", methods=["GET", "POST"])
@login_required
def consult():
    diagnosis = None
    follow_up_question = None

    if request.method == "POST":
        symptoms = request.form.get("symptoms")

        # Call the mock LLM to get diagnosis
        diagnosis = mock_llm(symptoms)

        # Generate follow-up question based on symptoms
        follow_up_question = further_questions(symptoms)

    return render_template("personal/consult.html", diagnosis=diagnosis, follow_up_question=follow_up_question)
