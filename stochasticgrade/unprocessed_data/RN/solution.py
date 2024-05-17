import random 

def main(): 
    numbers = []
    for i in range(10): 
        random_number = random.randint(1, 100) 
        numbers.append(random_number)
    return numbers
        
if __name__ == '__main__': 
    main()