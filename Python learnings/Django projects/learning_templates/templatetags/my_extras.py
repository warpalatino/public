
from django import template

register = template.Library()

#as customer filter, let's cut out all values of arg from the string
@register.filter
def cut(value, arg):
    return value.replace(arg, '')

#we could change the decorator with -- register.filter ('cut', cut) --