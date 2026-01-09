
def find_sets(cards): 
    '''
    Takes in an array of dictionaries representing cards and returns sets.
    '''

    return [(i, j, k) for i in range(len(cards) - 2) 
            for j in range(i + 1, len(cards) - 1) 
            for k in range(j + 1, len(cards)) 
            if is_set(cards[i], cards[j], cards[k])]


def is_set(card1, card2, card3): 
    for attr in ["color", "number", "shape", "fill"]:
        if len(set(c[attr] for c in [card1, card2, card3])) == 2: 
            return False
    return True