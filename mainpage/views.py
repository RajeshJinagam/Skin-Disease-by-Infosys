from django.shortcuts import render, redirect
from django.contrib.auth.forms import AuthenticationForm, UserCreationForm
from django.contrib.auth import authenticate, login
from django.contrib import messages
import base64
from django.core.files.storage import FileSystemStorage

def mainpage(request):
    return render(request, 'mainpage.html')

def loginpage(request):
    if request.method == "POST":
        if request.user.is_authenticated:
            return redirect('/profilepage')
        form = AuthenticationForm(data=request.POST)
        if form.is_valid():
            username = form.cleaned_data['username']
            password = form.cleaned_data['password']
            user = authenticate(request, username=username, password=password)
            if user is not None:
                login(request, user) 
                return redirect('/profilepage')
            else:
                messages.error(request, 'Invalid Username/Password')
        else:
            messages.error(request, 'Invalid form submission')
    form = AuthenticationForm()
    return render(request, 'loginpage.html', {'form': form})

def signuppage(request):  
    if request.method == "POST":
        form = UserCreationForm(request.POST)
        if form.is_valid():
            form.save()
            username = form.cleaned_data['username']
            password = form.cleaned_data['password1']  
            user = authenticate(username=username, password=password)
            if user is not None:
                login(request, user)  
                return redirect('/profilepage')
        else:
            messages.error(request, 'User registration failed. Please check your input.')
    else:
        form = UserCreationForm()

    return render(request, 'signuppage.html', {'form': form})

#def profilepage(request):
    if request.user.is_authenticated:
        message=""
        if request.method == "POST": 
            if request.FILES.get("image"):
                image_file = request.FILES["image"].read()
                encoded_image = base64.b64encode(image_file).decode('utf-8')
                img_data_url = f"data:image/jpeg;base64,{encoded_image}"
                return render(request, "profilepage.html", {"img": img_data_url})
            else:
                message = "Please select an image to upload."
        return render(request, "profilepage.html", {"message": message})
    return redirect("/loginpage")


def profilepage(request):
    if(request.method=="POST"):
        if(request.FILES.get('image')):
            img_name = request.FILES['image']
            fs = FileSystemStorage()
            filename = fs.save(img_name.name,img_name)
            img_url = fs.url(filename)
            return render(request,'profilepage.html',{'img':img_url})
    else:
        return render(request,'profilepage.html')

def about(request):
    return render(request,'about.html')