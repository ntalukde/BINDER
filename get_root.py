#!python3
"""
get_root.py
"""

def parse_file(datafile, sep=None):
    reader = open(datafile, "r")
    return tuple(
            tuple(line.strip().split(sep)) for line in reader
            if len(line) > 2 # avoids empty lines
    )

# only helps for pure numbers, i.e. still won't handle w10 vs w2 correctly
def number_safe_sort(x):
    if type(x) is str and x.isnumeric():
        return "{:015d}".format(int(x)) # 15 digits should be enough
    else:
        return x
        
def unpack(pairs, *, sort=True, numerical=False):
    # start with a set so there are no dupes
    if not sort:
        words_set = set()
        words = list()
        for (A, B) in pairs:
            if A not in words_set:
                words.append(A)
                words_set.add(A)
            if B not in words_set:
                words.append(B)
                words_set.add(B)
    else:
        words = set(pair[0] for pair in pairs).union(set(pair[1] for pair in pairs))
        # turn it into a list
        words = list(words)
        if numerical:
            words.sort(key=lambda w: int(w))
        else:
            words.sort(key=number_safe_sort)
    num_words = len(words)
    words_to_index = { words[i]: i for i in range(num_words) }
    pair_numbers = tuple( (words_to_index[w1], words_to_index[w2]) for (w1, w2) in pairs )

    return (words, num_words, words_to_index, pair_numbers)

def find_root(words, pairs):
    # a root object should be in the hypernym position (index 1) of every other word
    counts = {word: 0 for word in words}
    for pair in pairs:
        counts[pair[1]] += 1

    # everything except itself
    number_needed = len(words) - 1
    for word in words:
        if counts[word] >= number_needed:
            return word

    # there might not be a root
    return None


if __name__ == "__main__":
    import sys
    datafile = sys.argv[1].strip()
    
    pairs = parse_file(datafile, sep=',')
    # use sets to clean out duplicates, but convert to list at the end
    (words, num_words, word_to_index, pair_numbers) = unpack(pairs)
    # Find the root of the hierarchy, if there is one
    root_object = find_root(words, pairs)
    
    if root_object is not None:
       print("The root is:", root_object)
    else:
       print("This lattice has no root.")
    


