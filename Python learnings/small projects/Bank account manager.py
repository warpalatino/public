
#Bank Account Manager
#Create a class called Account which will be an abstract class for three other classes called CheckingAccount, 
#SavingsAccount and BusinessAccount. Manage credits and debits from these accounts through an ATM style program.

# -----------------

#activities for the ATM:
# 1 - starting screen saying hello 
# 2 - check credentials for log-in into profile
# 3 - select account to operate with
# 4 - present personalised initial data with options: 
#       * check full balance on each account
#       * move money from savings to checkings
#       * withdraw from either checking or business account
#       * execute a payment from checking or business account to whatever
#       * process money inflows
# 5 - Log out and greet


# Classes -----------------

class Account:
    def __init__(self, name):
        self.name = name



class CheckingAccount(Account):
    def __init__(self, name, c_balance = 1000):
        Account.__init__(self, name)
        self.c_balance = c_balance

    def c_balance(self):
        return self.c_balance



class SavingsAccount(Account):
    def __init__(self, name, s_balance = 2500):
        Account.__init__(self, name)
        self.s_balance = s_balance
    
    def s_balance(self):
        return self.s_balance



class BusinessAccount(Account):
    def __init__(self, name, b_balance = 5000):
        Account.__init__(self, name)
        self.b_balance = b_balance

    def b_balance(self):
        return self.b_balance



#Definitions -----------------------------------

name = Account('nico')
nico_c = CheckingAccount('nico')
nico_s = SavingsAccount('nico')
nico_b = BusinessAccount('nico')





# Log-in -----------------


def log_in(name):

    pin_code = 5555
    pin_security = []
        
    print('')
    print('')

    while True:
        
        inserted_pin_code = int(input('Checking your credentials. Please input your 4-digit PIN code: '))
        
        if inserted_pin_code == pin_code:

            print('')
            print('')
            print(f'Welcome {name} to your bank account.')
            pin_security = []
            break

        else:
                
            pin_security.append('x')

            if len(pin_security) == 1:
                
                print('')
                print('Your PIN code is incorrect, you have two more attempts left')  

            elif len(pin_security) == 2:

                print('')
                print('Your PIN code is incorrect again, you have only one more attempt left')

            elif len(pin_security) >= 3: 
                
                print('')
                print('Sorry the PIN code is wrong again. You have used three attemps.')
                print('The access to your account has been blocked.')
                print('Call customer service at 555 - 5555 - 5555.')
                pin_security = []
                break
        



# Opening menu -----------------

def initial_text(name):

    print('')
    print('')
    print('_  _  _  _  _')
    print('Please make a choice:')
    print('_  _  _  _  _')
    print('Option 1 - Go to your CHECKING account')
    print('Option 2 - Go to your SAVINGS account')
    print('Option 3 - Go to your BUSINESS account')
    print('')
    print('')






#Operations in the accounts -----------------------------------


def c_selection(name):
     
    while True:

        print('')
        print('')
        print('---CHECKING account---')
        print('What do you want to do: ')
        print('______________________')
        print('Option 4: deposit some money')
        print('Option 5: withdraw some money')
        print('Option 6: move some money to your Savings account')
        print('Option 7: make a transfer')
        print('Option 8: check your checking balance')
        print('Option 9: EXIT')
        print('______________________')
        print('')
        print('')    

        your_c_option = int(input('Select your option for the CHECKING account please - digit a number between 4 and 9: '))

        if your_c_option == 4:

            print('')
            print('')
            print(f'You have {nico_c.c_balance} available on the account')
            print('')
            c_deposit = int(input('How much do you want to deposit in your checking account? EUR'))
            nico_c.c_balance += c_deposit
            print('')
            print('')
            print('Please insert cash in the ATM...3...2...1...now.')
            print('Ok, done. Money received.')
            print(f'You have deposited EUR {c_deposit} into your checking account.')
            print(f'Your updated balance is now EUR {nico_c.c_balance}')
            
        elif your_c_option == 5:

            while True:

                print('')
                print('')
                print(f'You have {nico_c.c_balance} available on the account')
                print('')
                c_withdrawal = int(input('How much do you want to withdraw from your checking account? EUR'))
                        
                if c_withdrawal <= nico_c.c_balance and c_withdrawal > 0:
                    nico_c.c_balance -= abs(c_withdrawal)
                    print('')
                    print('')
                    print('Take the money.')
                    print(f'Your updated balance is now EUR {nico_c.c_balance}')
                    break
                                
                else:
                    print('')
                    print('[!!!]')
                    print('Sorry you do not have enough money avaible. Try with a smaller amount')
                    print('[!!!]')
                    break
                
                
        elif your_c_option == 6:

            print('')
            print('')
            print(f'You have {nico_c.c_balance} available on the account')
            print('')
            c_transfer_s = int(input('How much do you want to transfer from savings into CHECKING account? '))
            nico_c.c_balance -= c_transfer_s
            nico_s.s_balance += c_transfer_s
            print('')
            print('')
            print(f'Ok perfect, you are moving EUR {c_transfer_s} from your checking to your savings account.')
            print(f'Done. Money has been moved to your savings account')
            print(f'Your updated checking balance is now EUR {nico_c.c_balance} while your savings balance is now EUR {nico_s.s_balance}')
                
        elif your_c_option == 7:

            print('')
            print('')
            print(f'You have {nico_c.c_balance} available on the account')
            print('')
            c_transfer = int(input('How much do you want to transfer out to another bank account? EUR'))
            c_transfer_name = input('Add the recipient here')
            nico_c.c_balance -= c_transfer
            print('')
            print('')
            print(f'Operation completed. You are transferring out from your checking account EUR {c_transfer} to [{c_transfer_name}].')
            print(f'Your updated balance is now EUR {nico_c.c_balance}')
                        
        elif your_c_option == 8:

            print(f'your updated checking balance is EUR {nico_c.c_balance}')
                        
        elif your_c_option == 9:
                        
            print('Exit.')
            break

        else:

            print('Typo: please try again a number between 4 and 9 ')




def s_selection(name):
     
    while True:

        print('')
        print('')
        print('---SAVINGS account---')
        print('What do you want to do: ')
        print('______________________')
        print('Option 10: move some money to your checking account')
        print('Option 11: check your savings balance')
        print('Option 12: EXIT')
        print('______________________')
        print('')
        print('')
                
        your_s_option = int(input('Select your option for the SAVINGS account please - digit a number between 10 and 12: '))

        if your_s_option == 10:

            print('')
            print('')
            print(f'You have {nico_s.s_balance} available on the account')
            print('')
            s_transfer_c = int(input('How much do you want to transfer from CHECKING into SAVINGS? EUR'))
            nico_s.s_balance -= s_transfer_c
            nico_c.c_balance += s_transfer_c
            print(f'ok perfect, you are moving EUR {s_transfer_c} from your savings to your checking account.')
            print(f'Your updated savings balance is now EUR {nico_s.s_balance} while your checking balance is now EUR {nico_c.c_balance}')

        elif your_s_option == 11:

            print(f'your updated savings balance is EUR {nico_s.s_balance}')

        elif your_s_option == 12:

            print('Exit.')
            break

        else:

            print('Typo: please try again a number between 10 and 12 ')



def b_selection(name):
     
    while True:

        print('')
        print('')
        print('---BUSINESS account---')
        print('What do you want to do: ')
        print('______________________')
        print('Option 13: deposit some money')
        print('Option 14: withdraw some money')
        print('Option 15: make a transfer')
        print('Option 16: check your business balance')
        print('Option 17: EXIT')
        print('______________________')
        print('')
        print('')
        
        your_b_option = int(input('Select your option for the BUSINESS account please - digit a number between 13 and 17: '))

        if your_b_option == 13:

            print('')
            print('')
            print(f'You have {nico_b.b_balance} available on the account')
            print('')
            b_deposit = int(input('How much do you want to deposit in your business account? EUR'))
            nico_b.b_balance += b_deposit
            print('Please insert cash in the ATM...3...2...1...now.')
            print('Ok, done. Money received.')
            print(f'You have deposited EUR {b_deposit} into your business account.')
            print(f'Your updated balance is now EUR {nico_b.b_balance}')

        elif your_b_option == 14:
            
            print('')
            print('')
            print(f'You have {nico_b.b_balance} available on the account')
            print('')
            b_withdrawal = int(input('How much do you want to withdraw from your business account? EUR'))

            while True:

                if b_withdrawal <= nico_b.b_balance and b_withdrawal >0:
                    nico_b.b_balance -= abs(b_withdrawal)
                    print('')
                    print('')
                    print(f'Ok cool, you are withdrawing EUR {b_withdrawal} from your business account.')
                    print('Take the money.')
                    print(f'Your updated balance is now EUR {nico_b.b_balance}')
                    break
                                
                else:
                    print('')
                    print('[!!!]')
                    print('Sorry you do not have enough money avaible. Try with a smaller amount')
                    print('[!!!]')
                    break


        elif your_b_option == 15:

            print('')
            print('')
            print(f'You have {nico_b.b_balance} available on the account')
            print('')
            b_transfer = int(input('How much do you want to transfer out? '))
            b_transfer_name = input('Add the recipient here')
            nico_b.b_balance -= b_transfer
            print(f'Operation completed. You are transferring out from your business account EUR {b_transfer} to [{b_transfer_name}].')
            print(f'Your updated balance is now EUR {nico_b.b_balance}')

        elif your_b_option == 16:
                    
            print(f'your updated business balance is EUR {nico_b.b_balance}')
                
        elif your_b_option == 17:

            print('Exit.')
            break

        else:

            print('Typo: please try again a number between 13 and 17 ')

    

# Running the code after main menu -----------------


def continue_ATM(name):

    initial_text('nico')
    your_option = int(input('Digit a number between 1 and 3: '))
    print('----> Press 0 to exit immediately')

    while your_option >0 and your_option <=3:

        if your_option == 1:

            c_selection('nico')
            break

        elif your_option == 2:

            s_selection('nico')
            break
        
        elif your_option == 3:

            b_selection('nico')
            break



# Performing other ops from main menu -----------------


def replay_ATM(name):

    replay = input('Do you want to perform any other operation in any of the available accounts? [Y/N] ---> ')

    while True:

        if replay == 'Y':
            
            continue_ATM('nico')

        else:
                
            print('')
            print('')
            print('+++')
            print('Thank you very much and goodbye')
            print('+++')
            break
    





#Summary -----------------------------------

print('*****************************************')
print('      Welcome to your ATM machine')
print('*****************************************')
print('   ***********************************   ')
print('        *************************        ')
print('             ***************             ')
print('                  *****                  ')
print('')

log_in('nico')
continue_ATM('nico')
replay_ATM('nico')
        









    



