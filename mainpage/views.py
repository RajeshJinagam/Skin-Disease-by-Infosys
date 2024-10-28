from django.shortcuts import render, redirect
from django.contrib.auth.forms import AuthenticationForm, UserCreationForm
from django.contrib.auth import authenticate, login
from django.contrib import messages

def mainpage(request):
    return render(request, 'mainpage.html')

def loginpage(request):
    if request.user.is_authenticated:
        return redirect('/profilepage')
    if request.method == "POST":
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
    if request.user.is_authenticated:
        return redirect('/profilepage')  
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

def profilepage(request):
    if request.user.is_authenticated:
        return render(request, 'profilepage.html')  
    return redirect('/loginpage') 