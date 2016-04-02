from flask import Flask, request, render_template
from stitching import stitchImages, saveImage
app = Flask(__name__)

def processImages(request):
  filenames = []
  # Saves all images
  for name in request.files:
    image_file = request.files[name]
    if image_file :
      filenames.append(image_file.filename)
      image_file.save("images/" + image_file.filename)

  # Make the stitching
  result = stitchImages( filenames )
  saveImage("static/result.jpg", result)
  return render_template("result.html", result="result.jpg")

@app.route('/', methods=['GET', 'POST'])
def main():
    if request.method == 'POST':
        return processImages(request)
    else:
        return render_template( "index.html")

app.debug = True
app.run()
