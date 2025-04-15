from flask import Flask, request, jsonify
import pandas as pd
from flask_cors import CORS

# Import the question-based inference function from your inference script
# Make sure the file is named 'inference_dominantriats.py' and in the same folder
from inference_dominantriats import predict_personality

app = Flask(__name__)

CORS(app, resources={r"/predict": {"origins": [
                                                 "https://talentbridge-frontend.vercel.app",
                                                 
                                                 "https://prototype-tb-diegoschwape-diego-s-projects-a3f4cf0b.vercel.app/", 
                                                 "https://prototype-2tkcgj668-diego-s-projects-a3f4cf0b.vercel.app/", 
                                                 "https://*.vercel.app/", 
                                                 "https://talentbridge.io",
                                                 "https://www.talentbridge.io",
                                                 "https://prototype-tb.vercel.app",
                                                 "https://prototype-6oaduaro1-diego-s-projects-a3f4cf0b.vercel.app",
                                                 "https://prototype-tb-diego-s-projects-a3f4cf0b.vercel.app",
                                                 "https://prototype-tb-diegoschwape-diego-s-projects-a3f4cf0b.vercel.app"
                                                 ]}}, )


@app.route("/predict", methods=["POST"])
def predict_personality_endpoint():
    """
    Expects a JSON payload with up to 50 question strings mapped to integer scores (1–5).
    Example partial:
    {
      "I am the life of the party": 5,
      "I don't talk a lot": 2,
      "I feel comfortable around people": 4,
      ...
      "I am full of ideas": 4
    }

    Returns a JSON response with:
      "predictions": { "EXT": <float>, "AGR": <float>, ... },
      "dominant_traits": [ <top_trait>, <2nd_trait> ],
      "message": "Success"
    """
    try:
        # 1) Parse incoming JSON
        data = request.get_json(force=True)
        if not data:
            return jsonify({"error": "No JSON payload provided"}), 400

        # 2) Convert to a single-row DataFrame
        df = pd.DataFrame([data])

        # 3) Call your question-based inference function
        predictions_df, dominant_traits_list, processed_data = predict_personality(df)

        # predictions_df: columns [EXT, AGR, EST, CSN, OPN]
        # dominant_traits_list: e.g. [['EXT', 'OPN']] for the top 2 traits
        # processed_data: the final DataFrame with aggregated columns

        if len(predictions_df) == 0:
            return jsonify({"error": "No predictions returned"}), 500

        # For a single row, gather the results
        predictions = predictions_df.iloc[0].to_dict()
        top_two = dominant_traits_list[0] if dominant_traits_list else []

        # 4) Build JSON response
        response_data = {
            "predictions": predictions,
            "dominant_traits": top_two,
            "message": "Success"
        }
        return jsonify(response_data), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # Adjust host/port as needed
    app.run(host="0.0.0.0", port=8080, debug=True)
