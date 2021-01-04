
from django.shortcuts import render
#from django.http import HttpResponse --> not useful anymore if we use render
from ProTWO_app.models import User
from ProTWO_app.forms import NewUserForm



# Homepage view
def index(request):
    return render(request,'index.html')


# /users view (grabbing all user objects and returning them)
def users(request):
    user_list = User.objects.order_by('last_name')
    user_dict = {"users":user_list}
    return render(request,'users.html', context=user_dict)


#return information from the form: make an instance of the form class...
# ...pass the request method POST, check if valid, 
def my_model_form(request):

    form = NewUserForm()

    if request.method == 'POST':
        form = NewUserForm(request.POST)

        if form.is_valid():
            form.save(commit=True)
            return index(request)       #taking us back to the homepage

        else: 
            print("invalid form")

    return render(request, 'form.html', {'form': form})