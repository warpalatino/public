
import os
os.environ.setdefault('DJANGO_SETTINGS_MODULE','ProTWO.settings')

import django
django.setup()

from ProTWO_app.models import User
from faker import Faker
fakegen = Faker()

def populate(N=5):

    for entry in range(N):

        fake_name = fakegen.name().split()
        fake_first_name = fake_name[0]
        fake_last_name = fake_name[1]
        fake_email = fakegen.email()

        #create a new object in the database (tuple unpacking)
        User.objects.get_or_create(first_name=fake_first_name, last_name=fake_last_name, email=fake_email)[0]

#finally automatically populate our database via a for loop and N iterations
if __name__ == '__main__':
    print("Populating the databases...Please Wait")
    populate(20)
    print('Populating Complete')