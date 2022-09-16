from app import app
from flask import request, render_template
import os
import cv2
from PIL import Image
from skimage.metrics import structural_similarity

app.config["INITIAL_FILE_UPLOADS"] = 'app/static/uploads'
app.config['EXISTNG_FILE'] = 'app/static/original'
app.config['GENERATED_FILE'] = 'app/static/generated'


@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "GET":
        return render_template("index.html")

    if request.method == "POST":
        file_uploads = request.files['file_upload']
        filename = file_uploads.filename

        uploaded_image = Image.open(file_uploads).resize((250, 160))
        uploaded_image.save(os.path.join(app.config["INITIAL_FILE_UPLOADS"], "image.jpg"))

        original_image = Image.open(os.path.join(app.config['EXISTNG_FILE'], "image.jpg")).resize((250, 160))
        original_image.save(os.path.join(app.config['EXISTNG_FILE'], "image.jpg"))

        original_image = cv2.imread(os.path.join(app.config['EXISTNG_FILE'], "image.jpg"))
        uploaded_image = cv2.imread(os.path.join(app.config["INITIAL_FILE_UPLOADS"], "image.jpg"))

        original_gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
        uploaded_gray = cv2.cvtColor(uploaded_image, cv2.COLOR_BGR2GRAY)

        (score, diff) = structural_similarity(original_gray, uploaded_gray, full=True)
        diff = (diff * 255).astype("uint8")

        _, thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        cnts, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for c in cnts:
            (x, y, w, h) = cv2.boundingRect(c)
            cv2.rectangle(original_image, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.rectangle(uploaded_image, (x, y), (x + w, y + h), (0, 0, 255), 2)

        cv2.imwrite(os.path.join(app.config['GENERATED_FILE'], 'image_original.jpg'), original_image)
        cv2.imwrite(os.path.join(app.config['GENERATED_FILE'], 'image_uploaded.jpg'), uploaded_image)
        cv2.imwrite(os.path.join(app.config['GENERATED_FILE'], 'image_diff.jpg'), diff)
        cv2.imwrite(os.path.join(app.config['GENERATED_FILE'], 'image_thresh.jpg'), thresh)
        return render_template("index.html", pred=str(round(score * 100, 2)) + "% correct")


if __name__ == '__main__':
    app.run(debug=True)
