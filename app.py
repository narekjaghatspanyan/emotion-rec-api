import os
from flask import Flask, render_template, request, jsonify
from detect_emotions import detect_emotion

app = Flask(__name__)
app.config["IMAGE_UPLOADS"] = ""
port = int(os.environ.get("PORT", 5100))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/emotion_recognizer', methods = ['GET', 'POST'])
def recocnize_faces():
   if request.method == 'POST':
       image = request.files['image']
       image.save(os.path.join(app.config["IMAGE_UPLOADS"], image.filename))
       response_emotion = detect_emotion(image.filename)
       return jsonify(
           resp_emotion=response_emotion)

if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0',port=port)
