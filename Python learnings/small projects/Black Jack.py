#Black Jack game with OOP
#one computer dealer and one human player
#keep track of money
#three scenarios: human player goes bust (over 21), dealer wins (17-21), dealer goes bust

#LOGIC

#Create a card, and a deck of 52 cards (with random shuffle)
#Create a hand and chips
#Ask the Player for their bet (no more than available chips)
#Deal two cards to the Dealer (show one only) and two cards to the Player (show both)
#Ask the Player if they wish to Hit, and take another card (loop, and always check that they do not go over 21)
#If a Player Stands, play the Dealer's hand. 
#The dealer will always Hit until the Dealer's value meets or exceeds 17
#Determine the winner and adjust the Player's chips accordingly
#Ask the Player if they'd like to play again

#-----------------------

#START: classes

#Assign global variables to describe deck of cards
#Then compose a single CARD with a class
#Then compose the DECK by leveraging card attributes and start adding functions
#Suits and ranks are tuples (immutable), while value of each card is in a dictionary
#We also create a HAND class to assign card values and a CHIPS class to manage bets
#import random library to shuffle deck later on



import random

suits = ('Hearts', 'Diamonds', 'Spades', 'Clubs')
ranks = ('Ace', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight', 'Nine', 'Ten', 'Jack', 'Queen', 'King')
values = {'Two':2, 'Three':3, 'Four':4, 'Five':5, 'Six':6, 'Seven':7, 'Eight':8, 'Nine':9, 'Ten':10, 'Jack':10,
         'Queen':10, 'King':10, 'Ace':11}

#we type booleans above here to control while loops



#-------



class Card():

    def __init__(self, suit, rank):
        self.suit = suit    #we leave the tuple above with suits, while we call the attribute suit
        self.rank = rank    #we leave the tuple above with ranks, while we call the attribute rank

    def __str__(self):
        return f'{self.rank} of {self.suit}'



#-------



class Deck():

    def __init__(self):
        self.deck = []  
        for suit in suits:
                    for rank in ranks:      
                        Card(suit,rank)    
                        self.deck.append(Card(suit,rank))  

    def __str__(self):
        deck_comp = ''  
        for card in self.deck:  
            deck_comp += '\n' + str(card)   
        return ('The deck has:' + deck_comp)

    def shuffle(self):
        random.shuffle(self.deck)

    def deal(self):
        single_card = self.deck.pop()
        return single_card



#__init__
#deck has no attribute, as we want it to be different all the times we play
#no additional attributes for deck: we pre-defined Card's attributes above, and we bring them down here to the deck class
#we need two nested for loops to iterate on two attributes
#we built a Card object and so we use such OOP wording Card(suit,rank) (but classic variable card = suit, rank) works
#we use append instead of += (similar) as classes per se are not iterable

#__str__
#starting by an empty string will allow concatenation, so that we can then add all individual card names (strings) to describe deck
#we cannot iterate on class Card directly (nor assign class to function x = Card(suit, rank)...
# ... so we iterate on whatever variable we want - i.e. card, or x - in the entire self.deck which was instantiated above
#even within the string __str__ definition, we have to make clear that card will have to be a string itself
#instead of deck_comp += str(card) we could also write deck_comp += card.__str__()

#shuffle
#classic shuffle function to mix the deck
#shuffling does not require 'return'

#deal
#instructions require in deal to hand over a card each time we play a round, and then pop/remove a card from the deck
#it is kind-of the perspective of the dealer that sees the deck going down in size
#we could have summarized it in one line with return = self.deck.pop()

#>>>>>>>>>>>>>
#Test here below to check if we have the entire 52 cards deck
# --> test_deck = Deck()
# --> test_deck.shuffle = Deck()
# --> print(test_deck)
#>>>>>>>>>>>>>



#-------



class Hand():

    def __init__(self):
        self.cards = []  
        self.value = 0   
        self.aces = 0    

    def add_card(self,card):
        self.cards.append(card) 
        self.value += values[card.rank]
        if card.rank == 'Ace':      
            self.aces += 1

    def adjust_for_ace(self):           
            while self.value > 21 and self.aces > 0:
                self.value -= 10
                self.aces =-1


#__init__
#the hand class is basically a representation of a round of the player/dealer
#we create an empty list of cards handed over to a player (self.cards) where we add cards from deck (self.deck.pop)
#self.value (of cards in hand) start with zero, and it will become the sum of each player/dealer's hand
#instructions require to add an attribute to keep track of aces, and also here we start from zero

#add_card
#this is a key function to monitor/sum the cards handed over, and check their value
#notice that cards will come from dealer's deck, so we should link add_card() to single_card, or to the function deal (=deck.deal())
#we retrieve dictionary values by typing my_dictionary[key1] = value1 (which is the standard)...
# ... so here we have to use 'values[ranks]' where ranks is the attribute rank inside class Card, or...card.rank

#adjust for aces
#as per instructions, we consider Aces a special card and we start counting them when they appear
#as per instructions, we need to adjust for Aces to reduce the value from 11 to 1 when we exceed 21 with the first two cards
#we could write while self.value > 21 and self.aces > 0: or while self.value > 21 and self.aces: as it is the sawme thing


#>>>>>>>>>>>>>
# we assign variables, activate the deal function to pull card from deck (2x), activate the add_card to hand cards, then show/count 
# --> test_deck = Deck()
# --> test_deck.shuffle()
# --> test_player = Hand()
# --> pulled_card0 = test_deck.deal()
# --> pulled_card1 = test_deck.deal() 
# --> test_player.add_card(pulled_card0)
# --> print(pulled_card0)
# --> test_player.add_card(pulled_card1)
# --> print(pulled_card1)
# --> test_player.value
# --> print(test_player.value)
#>>>>>>>>>>>>>



#-------



class Chips():

    def __init__(self, total = 100):
        self.total = total
        self.bet = 0 #small list to sum up the bet at each round

    def win_bet(self):
        self.total += self.bet

    def lose_bet(self):
        self.total -= self.bet


#__init__
# I can give an attribute to chips (=how many they are, or total), and define it as usual 'self.total = total' 
# but we can also assign a number total_chips = 100, random, as starting value
# alternatively, we could have no attributes and immediately define self.total = 100


#-------



#CONTINUE: operate game via functions



#-------



#A. take bets before starting a round

def take_bet(chips):
    while True:
        try:
            chips.bet = int(input('How much money would you like to bet this round? '))
            print(f"Thanks for your bet of {player_chips.bet}. Let's start.")
            print("      ")
            print("   +++   ")  
        except:
            print ('Sorry, only rounded numbers allowed')
        else:
            if chips.bet > chips.total:
                print(f'Sorry, you do not have enough chips, as you have only {chips.total}')
            else:
                break

#we have a function take_bet to which we assign the variable chips, which we want to determine. As we have a self.total and self.bet attributes in the Chips class...
# ... we simply transform self.total from Chips into chips.total, and self.bet from Chips into chips.total



#B. Handle every win or lose situation


#C. Show some/all cards

def show_some(player,dealer):
    print("\nDealer's Hand:")
    print("<card hidden>")
    print('',dealer.cards[1])  
    print("\nPlayer's Hand:", *player.cards, sep='\n ')
    
def show_all(player,dealer):
    print("\nDealer's Hand:", *dealer.cards, sep='\n ')
    print("Dealer's Hand value = ",dealer.value)
    print("\nPlayer's Hand:", *player.cards, sep='\n ')
    print("Player's Hand value = ",player.value)




#-------



#RUN THE GAME



#-------

playing = True


while True:
    
    print("      ")
    print("Hey player, welcome to Nico's black jack: an amazing project!")
    print("We are preparing the cards' deck, and shuffling it. ")
    print("Get ready.")
    print("   ____   ")
    print("      ")

    # Create & shuffle the deck and deal two cards to the player
    nicos_deck = Deck()
    nicos_deck.shuffle()
    nicos_deck.deal()
    nicos_deck.deal()

    # Create a player's hand by adding the two cards
    player_hand = Hand()
    player_hand.add_card(nicos_deck.deal())
    player_hand.add_card(nicos_deck.deal())
    print("Ok player: you now have two cards")
    
    # Do the same for the dealer
    dealer_hand = Hand()
    nicos_deck.deal()
    nicos_deck.deal()
    dealer_hand.add_card(nicos_deck.deal())
    dealer_hand.add_card(nicos_deck.deal())
    print("Dealer has two cards too")
    print("   ____   ")
    print("      ")

    # Set up the player's chips and prompt the player for their bet
    player_chips = Chips()
    take_bet(player_chips)

    # Show cards (but keep one dealer card hidden)
    show_some(player_hand,dealer_hand)
    print(player_hand.value)

    # Player plays
    while playing:  
    
        ask = str(input('Would you like to hit or stand? [H/S] '))
        
        if ask.lower() == 'h':
            
            player_hand.add_card(nicos_deck.deal())  #we deliver it to the player
            player_hand.adjust_for_ace()       #we adjust for ace
            show_some(player_hand,dealer_hand)
            print(player_hand.value)
            print('_______________')
            print(' ')

        elif ask.lower() == 's':
                print("Ok, you stand. Dealer: your move.")
                #show_some(player_hand,dealer_hand)
                #print(player_hand.value)
                print('_______________')
                print(' ')
            
        else: 
            print ('Sorry, typo: please try again typing only H or S')
            continue

        if player_hand.value > 21:
            print('Player bust, Dealer wins')
            player_chips.lose_bet()
            print(f'Player, your chips are now {player_chips.total}')
            break

        elif player_hand.value <= 21 and ask.lower() == 's':
            
            while dealer_hand.value < 17:
                
                print(' ')
                print('Dealer takes a card')
                dealer_hand.add_card(nicos_deck.deal())  #we deliver it to the player
                dealer_hand.adjust_for_ace()       #we adjust for ace
                show_all(player_hand,dealer_hand)
                break
                
            if dealer_hand.value > 17 and dealer_hand.value <=21:
                print(' ')
                print('Dealer stands.')
                print("Let's take a look at the cards again")
                show_all(player_hand,dealer_hand)
                print('_______________')

                if player_hand.value > dealer_hand.value:
                    print(' ')
                    print('Player wins')                
                    player_chips.win_bet()
                    print(f'Player, your chips are now {player_chips.total}')
                    break

                elif dealer_hand.value > player_hand.value:
                    print(' ')
                    print('Dealer wins')
                    player_chips.lose_bet()
                    print(f'Player, your chips are now {player_chips.total}')
                    break

                else:
                    print(' ')
                    print("It's a tie, no chips' exchange!")
                    print(f'Player, your chips remain {player_chips.total}')
                    break    
                
            elif dealer_hand.value > 21:
                print(' ')
                print('Dealer bust, Player wins')
                player_chips.win_bet()
                print(f'Player, your chips are now {player_chips.total}')
                break

    replay = input('Dear player, do you want to play another round?[Y/N] ')
        
    if replay.lower() == 'y':
        playing = True
        player_hand = True
        dealer_hand = True
        print("          ")
        print("   ****   ")
        print("   NEW GAME   ")
        print("   ****   ")
        print("          ")

    else:
                print("Ok, no worries. We finish here then. See you next time.")
                playing = False

