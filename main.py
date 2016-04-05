from flask import Flask, request, render_template
from stitching import stitchImages, saveImage
from nocache import nocache
import uuid
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
  result_filename = str(uuid.uuid1())
  print ("Result name: " + result_filename)
  saveImage("static/"+(result_filename)+".jpg", result)
  return render_template("result.html", result= (result_filename + ".jpg") )

@app.route('/', methods=['GET', 'POST'])
@nocache
def main():
    if request.method == 'POST':
        return processImages(request)
    else:
        return render_template( "index.html")

app.debug = True
app.run()
