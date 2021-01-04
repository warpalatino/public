
#Caesar cipher - Implement a Caesar cipher, both encoding and decoding. 
#The key is an integer from 1 to 25. This cipher rotates the letters of the alphabet (A to Z). 
#The encoding replaces each letter with the 1st to 25th next letter in the alphabet (wrapping Z to A). 
#So key 2 encrypts "HI" to "JK", but key 20 encrypts "HI" to "BC". This simple "monoalphabetic substitution cipher" 
#provides almost no security, because an attacker who has the encoded message can either use frequency analysis to guess 
#the key, or just try all 25 keys. 

# ------------------


#importing the entire alphabet (in uppercase) and naming it to use it (=my_alphabet)

import string 
alphabet = string.ascii_uppercase
my_alphabet = alphabet


#creating a class with some specific functions...

class Caesar():

    def __init__(self):
        self.letters = alphabet
        self.changed_message = ''

    def crypt(self, mode):                                  #defining the function that will change our message in the encryption process, to be then used for encryption or decryption
        
        for x in self.message.upper():                      #for any letter contained in the provided message (transformed in uppercase to match with alphabet)...

            if x in self.letters:                           #if each specific letter is in the alphabet (and is not a number for example)....
                num = self.letters.find(x)                  #then find the index position of each individual letter in the alphabet 
                
                if mode == 'E':                             #and then...that individual index position (num) will add a key to itself and change (if we have to encrypt the message)...
                    num = num + self.key
                elif mode == 'D':                           #...or it will detract a key to itself (if we have to decrypt the message)
                    num = num - self.key

                if num >= len(self.letters):                #and as we have only 25 letters, if any resulting num+key goes over...we will restart the count to stay within [1 - 25] range...
                    num = num - len(self.letters)
                elif num < 0:                               #hygiene check (I think this is just a hygiene check as we cannot have a negative index number in the alphabet...)
                    num = num + len(self.letters)

                self.changed_message += self.letters[num]   #this is the final result: we create a new string with each letter being the letter at the new index point from our message
            
            else:
                self.changed_message += x                   #any element in our message which is not a letter (such as a number or symbol) will be added to the new string as it is

        return self.changed_message                         #whatever the condition in the loop and subsequent formulas, return the newly resulted string with the newly indexed letters 

    def encrypt(self, message, key=0):                      #defining here the process of encryption by leveraging the crypt function (under condition 'E')
        self.changed_message = ''
        self.key = key
        self.message = message
        return self.crypt('E')

    def decrypt(self, message, key=0):                      #defining here the process of decryption by leveraging the crypt function (under condition 'D')
        self.changed_message = ''
        self.key = key
        self.message = message
        return self.crypt('D')

#running code ----------------------

#if __name__ == '__main__':
cipher = Caesar()
message = input('What is the text that you want to transform: ')
key = int(input('Select a numerical key between 1 and 25: '))
mode = input('Do you want to encrypt or decrypt? Input E or D only ')

def choice(mode):

    if mode == 'E':

        cipher.encrypt(message, key)
        print(cipher.encrypt(message, key))

    elif mode == 'D':

        cipher.decrypt(message, key)
        print(cipher.decrypt(message, key))

    else:

        print('Sorry wrong input. Try again.')

choice(mode)