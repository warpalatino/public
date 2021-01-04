from django.db import models
from django.contrib.auth.models import User
# Which data the user already has:
# SuperUserInformation
# User: Jose
# Email: training@pieriandata.com
# Password: testpassword

# Create your models here.
class UserProfileInfo(models.Model):

    # Create relationship (don't inherit from User!)
    user = models.OneToOneField(User, on_delete=models.CASCADE)

    # Add any additional attributes to the user you want
    portfolio_site = models.URLField(blank=True)

    # pip install pillow to use this, so that users do not need to upload their pic if they
    #...do not want it
    profile_pic = models.ImageField(upload_to='basic_app/profile_pics',blank=True)

    def __str__(self):
        # Built-in attribute of django.contrib.auth.models.User !
        return self.user.username
