from flask import Flask, request, render_template
from stitching import stitchImages, saveImage
from nocache import nocache
import uuid
app = Flask(__name__)

def loadAndSaveImage(image_file):
    if image_file :
      image_file.save("images/" + image_file.filename)
    return image_file.filename

def processImages(request):
  filenames = []
  # Saves all images
  filenames.append(loadAndSaveImage(request.files["01"]))
  filenames.append(loadAndSaveImage(request.files["02"]))
  filenames.append(loadAndSaveImage(request.files["03"]))

  filenames = filter(None, filenames)

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
