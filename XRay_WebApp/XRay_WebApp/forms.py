from django import forms

class UploadForm(forms.Form):
  image1 = forms.ImageField()
  image2 = forms.ImageField()
  history = forms.CharField(max_length=1024)

