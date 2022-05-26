from django.shortcuts import render
from Backend.cap_model import Model
from .forms import UploadForm
from PIL import Image
from django.views.decorators.csrf import csrf_exempt
import PIL 
import base64

@csrf_exempt 
def home(request):
    
    result = None
    form = None
    history = None
    image1_uri = None
    image2_uri = None

    if request.method == 'POST':
        form = UploadForm(request.POST, request.FILES)
        if form.is_valid():
            image1 = form.cleaned_data['image1']
            image2 = form.cleaned_data['image2']
            history = form.cleaned_data['history']

            image1_bytes = image1.file.read()
            encoded_img1 = base64.b64encode(image1_bytes).decode('ascii')
            image1_uri = 'data:%s;base64,%s' % ('image1/jpeg', encoded_img1)

            image2_bytes = image2.file.read()
            encoded_img2 = base64.b64encode(image2_bytes).decode('ascii')
            image2_uri = 'data:%s;base64,%s' % ('image2/jpeg', encoded_img2)

            try:
              result = getPredictions('/content/X_GEN/XRay_WebApp/Backend/input/', history, image1, image2)
            except RuntimeError as re:
              print(re)
    else:
        form = UploadForm()
    context = {
        'form': form,
        'result': result,
        'history': history,
        'image1_uri': image1_uri,
        'image2_uri': image2_uri,
    }
    return render(request, 'index.html', context)

def getPredictions(directory,medical_history, image0, image1):
    model = Model(directory, image0, image1)
    rep = model.get_report(medical_history)
    return rep
