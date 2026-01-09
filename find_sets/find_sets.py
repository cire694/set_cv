
def find_sets(cards): 
    '''
    Takes in an array of dictionaries representing cards and returns sets represented by triplets of indices.
    '''

    return [(i, j, k) for i in range(len(cards) - 2)
        for j in range(i + 1, len(cards) - 1)
        for k in range(j + 1, len(cards))
        if
            all(len(set(c[attr] for c in [cards[i], cards[j], cards[k]])) != 2
            for attr in ["color", "number", "shape", "fill"])
        ]
