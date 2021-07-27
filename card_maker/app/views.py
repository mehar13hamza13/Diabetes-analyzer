from django.shortcuts import render, redirect, get_object_or_404
from django.http import StreamingHttpResponse
from django.shortcuts import render
from django.contrib.auth.models import User
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth import login
from django.contrib.auth.decorators import login_required
from .forms import CreateUserForm,UserProfileForm, DashboardForm
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.hashers import  check_password, make_password
from django.contrib.auth import login as auth_login
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from . models import Dashboard
from django.core.paginator import Paginator, EmptyPage, PageNotAnInteger
import pandas as pd
import numpy as np
import pickle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier

# Create your views here.

def home(request):
    if request.method=="POST":
        request.session.flush()

        username = request.POST["username"]
        password = request.POST["password"]

        user = User.objects.filter(username__iexact=username).exists()

        if user:
            user = User.objects.filter(username__iexact = request.POST["username"])[0]
            if check_password(password,user.password):
                request.session['logged_in'] = {'username':user.username, 'id':user.id}

                return redirect('new')


        context = {

            'no_match':True
        }

        return render(request, 'index.html', context)


    return render(request, 'index.html')

#-------loading models-------
svc = pickle.load(open('app/models/SVC.sav', 'rb'))
knn = pickle.load(open('app/models/KNN.sav', 'rb'))
naivebayes = pickle.load(open('app/models/naivebayes.sav', 'rb'))
randomforest = pickle.load(open('app/models/RandomForest.sav', 'rb'))
ensemble = pickle.load(open('app/models/ensemble.sav', 'rb'))

def new(request):
    if 'logged_in' in request.session:
        return render(request, 'new.html')

    else:
        return redirect('signup')

def results(request):
    if request.method == "POST":
        print(request.POST)
        data = []
        pregnancy = int(request.POST['pregnancy'])
        glucose = int(request.POST['glucose'])
        bloodPressure = int(request.POST['bloodPressure'])
        skinThickness = int(request.POST['skinThickness'])
        insuline = int(request.POST['insulin'])
        bmi = int(request.POST['bmi'])
        diabetes = int(request.POST['diabetes'])
        age = int(request.POST['age'])

        data.append(pregnancy)
        data.append(glucose)
        data.append(bloodPressure)
        data.append(skinThickness)
        data.append(insuline)
        data.append(bmi)
        data.append(diabetes)
        data.append(age)

        data = np.array(data)
        data = data.reshape(1, 8)

        predictions = []
        print(data)
        predictions.append(svc.predict(data)[0])
        predictions.append(knn.predict(data)[0])
        predictions.append(naivebayes.predict(data)[0])
        predictions.append(randomforest.predict(data)[0])
        predictions.append(ensemble.predict(data)[0])
        print(predictions)
        result = predictions.count(0) if predictions.count(0) > predictions.count(1) else predictions.count(1)
        return render(request, 'results.html', {'result': result})
    else:
        return redirect('new')
def dashboard(request):
    if 'logged_in' in request.session:
        dashboard = Dashboard.objects.filter(user=request.session['logged_in']['id']).order_by('-id')
        page = request.GET.get('page', 1)
        paginator = Paginator(dashboard, 1)
        try:
            dashboard = paginator.page(page)
        except PageNotAnInteger:
            users = paginator.page(1)
        except EmptyPage:
            users = paginator.page(paginator.num_pages)

        return render(request, 'dashboard.html',{'dashboard':dashboard})
    else:
        return redirect('signup')


def checkout(request):
    return render(request, 'checkout.html')


def signupUser(request):

    if 'logged_in' in request.session:
        return render(request, 'new.html')
    else:
        form = CreateUserForm()
        request.session.flush()
        if request.method == 'POST':
            user = {'username':request.POST['username'], 'email':request.POST['email'], 'password1':request.POST['password'], 'password2':request.POST['password']}
            print(user)
            form = CreateUserForm(user)
            if form.is_valid():
                form.save()
                user = User.objects.all().order_by('-id')[0]
                form = UserProfileForm({'address':request.POST['address'], 'DOB':request.POST['DOB'], 'user':user})
                if form.is_valid():
                    instance = form.save()
                    messages.success(request, 'Account was created for ' + user.username)
                    request.session['logged_in'] = {'username':user.username, 'id':user.id}
                    return redirect('home')
                else:
                    print(form.errors)

            else:
                print(form.errors)


        context = {'form':form}
        return render(request, 'signup.html', context)

def loginUser(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')

        user = authenticate(request, username=username, password=password)

        if user is not None:
            login(request, user)
            request.session['logged_in'] = {'user_id':user.id}
            return redirect('home')
        else:
            messages.info(request, 'Username or Password is Incorrect')
            # return render(request, 'login.html', Context)

    Context = {}
    return redirect('home')


def logoutUser(request):
    logout(request)
    return redirect('home')