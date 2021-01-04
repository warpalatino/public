from django.shortcuts import render
from django.urls import reverse_lazy
from django.http import HttpResponse
from django.views.generic import (View,TemplateView,ListView,DetailView,
                                CreateView,DeleteView,UpdateView)
from basic_app import models

# Create your views here.

# Original Function View:
# def index(request):
#     return render(request,'index.html')

# A class-based new view to substitute the original function:
class CBView(View):
    def get(self,request):
        return HttpResponse('Class Based Views are Cool!')

# Now moving to model views...
class IndexView(TemplateView):
    template_name = 'index.html'            
    # template_name = 'app_name/site.html' if html file is in different directory

    def get_context_data(self,**kwargs):
        context  = super().get_context_data(**kwargs)
        context['injectme'] = "Basic Injection!"
        return context

class SchoolListView(ListView):
    model = models.School
    # this class automatically creates a list that can be called with school_list
       

class SchoolDetailView(DetailView):
    context_object_name = 'school_details'
    model = models.School
    template_name = 'basic_app/school_detail.html'


class SchoolCreateView(CreateView):
    fields = ("name","principal","location")
    model = models.School


class SchoolUpdateView(UpdateView):
    fields = ("name","principal")
    model = models.School

class SchoolDeleteView(DeleteView):
    model = models.School
    success_url = reverse_lazy("basic_app:list")



