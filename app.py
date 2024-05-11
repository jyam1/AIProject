from flask import Flask, render_template, request
from transformers import GPT2LMHeadModel, GPT2Tokenizer

app = Flask(__name__)

# Load pre-trained model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

def generate_response(task, limitations):
    prompt = f"I need to {task} with these limitations: {limitations}. Please make me a list of steps to complete this efficiently."
    
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    
    output = model.generate(input_ids, max_length=100, num_return_sequences=1)
    
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    
    steps = generated_text.split("\n")
    return steps

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/generate", methods=["POST"])
def generate():
    task = request.form.get("task")
    limitations = request.form.get("limitations")
    steps = generate_response(task, limitations)
    return render_template("response.html", task=task, limitations=limitations, steps=steps)

if __name__ == "__main__":
    app.run(debug=True)
