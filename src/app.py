from flask import Flask, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Cargar modelo y tokenizer
pipe = pipeline("text-generation", model="openai-community/gpt2")
tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")

if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

# Mapeo de equivalencias de CO₂
def get_co2_equivalent(total_pollution):
    equivalences = [
        (0.0001, "Encender una cerilla"),
        (0.0005, "Buscar una palabra en Google"),
        (0.001, "Abrir una página web básica"),
        (0.002, "Ver un correo electrónico durante 10 segundos"),
        (0.003, "Escuchar un segundo de música"),
        (0.004, "Enviar una postal física por correo"),
        (0.005, "Encender una bombilla LED durante 10 segundos"),
        (0.006, "Leer un tweet con una imagen adjunta"),
        (0.007, "Enviar un mensaje con emoji"),
        (0.008, "Escribir un comentario en redes sociales"),
        (0.009, "Enviar un correo electrónico"),
        (0.01, "Hervir agua"),
        (0.02, "Encender un coche"),
        (0.03, "Realizar una búsqueda y leer 3 resultados en Google"),
        (0.04, "Ver un minuto de video en calidad estándar"),
        (0.05, "Participar en una videollamada de 5 segundos"),
    ]
    for limit, activity in equivalences:
        if total_pollution <= limit:
            return activity
    return "Actividad fuera del rango especificado"

@app.route("/generate", methods=["POST"])
def generate():
    try:
        data = request.json
        user_message = data.get("message", "")

        prompt = f"User:{user_message} Bot:"
        input_ids = tokenizer.encode(prompt, return_tensors="pt")

        max_input_tokens = 1024 - 50
        if len(input_ids[0]) > max_input_tokens:
            return jsonify({"error": "The message is too long to process."})

        output = model.generate(
            input_ids,
            max_length=min(len(input_ids[0]) + 100, 1024),
            pad_token_id=tokenizer.eos_token_id,
            no_repeat_ngram_size=2,
            temperature=0.7,
            top_p=0.9,
            do_sample=True
        )

        bot_response = tokenizer.decode(output[:, input_ids.shape[-1]:][0], skip_special_tokens=True).strip() if output.size(1) > input_ids.size(1) else "Sorry, I was unable to generate a response."

        user_length = len(user_message)
        bot_length = len(bot_response)
        user_pollution = user_length * 0.00001
        bot_pollution = bot_length * 0.00001

        total_pollution = user_pollution + bot_pollution

        user_pollution_str = f"Your message weighs approximately ~{user_length * 0.01:.3f} KB, contaminating some (~{user_pollution:.6f} g de CO₂)"
        bot_pollution_str = f"My answer weighs approx ~{bot_length * 0.01:.3f} KB, contaminating some (~{bot_pollution:.6f} g de CO₂)"

        co2_equivalent = get_co2_equivalent(total_pollution)

        return jsonify({
            "response": bot_response,
            "user_pollution": user_pollution_str,
            "bot_pollution": bot_pollution_str,
            "total_consumed": total_pollution,
            "co2_equivalent": co2_equivalent
        })

    except Exception as e:
        return jsonify({"error": f"Error procesando la solicitud: {str(e)}"})

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
