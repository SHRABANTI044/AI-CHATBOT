from flask import Flask, render_template, request, jsonify
import os
from dotenv import load_dotenv
from google import genai
from google.genai import errors, types

load_dotenv()

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"error": "Request body must be valid JSON."}), 400

    user_message = data.get("message", "")
    if not isinstance(user_message, str) or not user_message.strip():
        return jsonify({"error": "Message must be a non-empty string."}), 400

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        app.logger.error("GEMINI_API_KEY is not configured.")
        return jsonify({"error": "Gemini API key is not configured on the server."}), 503

    try:
        client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(
                timeout=20000,
                retry_options=types.HttpRetryOptions(attempts=1),
            ),
        )
        model = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")
        fallback_models = [
            candidate.strip()
            for candidate in os.getenv(
                "GEMINI_FALLBACK_MODELS",
                "gemini-3-flash-preview",
            ).split(",")
            if candidate.strip()
        ]
        fallback_models = list(dict.fromkeys(fallback_models))
        try:
            response = client.models.generate_content(
                model=model,
                contents=user_message.strip(),
            )
        except errors.APIError as exc:
            if exc.code not in (429, 499, 503, 504):
                raise
            fallback_error = exc
            for fallback_model in fallback_models:
                if fallback_model == model:
                    continue
                app.logger.warning(
                    "Gemini model %s is unavailable; trying fallback model %s.",
                    model,
                    fallback_model,
                )
                try:
                    response = client.models.generate_content(
                        model=fallback_model,
                        contents=user_message.strip(),
                    )
                    break
                except errors.APIError as fallback_exc:
                    if fallback_exc.code not in (429, 499, 503, 504):
                        raise
                    fallback_error = fallback_exc
            else:
                raise fallback_error

        reply = response.text
        if not reply:
            app.logger.error("Gemini returned an empty response.")
            return jsonify({"error": "Gemini returned an empty response."}), 502
        return jsonify({"reply": reply})
    except errors.APIError as exc:
        app.logger.warning("Gemini API returned HTTP %s: %s", exc.code, exc.message)
        status = exc.code if exc.code in (429, 499, 503, 504) else 502
        return jsonify({"error": f"Gemini API error ({exc.code}): {exc.message}"}), status
    except Exception:
        app.logger.exception("Gemini chat request failed.")
        return jsonify({"error": "Gemini could not process the request. Check the server logs."}), 502

if __name__ == "__main__":
    app.run(debug=True) 