import difflib
import editdistance
import string
import re
from collections import Counter
from spellchecker import SpellChecker
from collections import Counter
from sklearn.datasets import fetch_20newsgroups
import re

dictionary = set()

def read_dictionary_file():
    global dictionary
    if dictionary:
        return
    with open("linuxwords.txt", "r") as f:
        contents = f.read()
    dictionary = set(
        word.lower()
        for word in contents.splitlines()
    )

def word_check(s):
    read_dictionary_file()
    for word in s:
        if word not in dictionary:
            print(word + ": spelt incorrectly")
            suggestion = difflib.get_close_matches(word, dictionary)
            print("Did you mean:")
            for x in suggestion:
                # calculate Levenshtein distance
                distance = editdistance.eval(x, word)
                print('%s , %d' % (x, distance))


def word_check2(s):
    spell_corrector = SpellChecker()
    for word in s:
        # Get the one `most likely` answer
        if word != spell_corrector.correction(word):
            print("Word incorrectly spelt:" + word)
            print("Most likely: " + spell_corrector.correction(word))
            # Get a list of `likely` options
            print("Other possibilities: " + str(spell_corrector.candidates(word)))

def remove_punctuation(s):
    # define punctuation
    punctuation = '''''!()-[]{};:'"\,<>./?@#$%^&*_~'''
    no_punct = ""
    for char in s:
        if char not in punctuation:
            no_punct = no_punct + char
            # display the unpunctuated string
    print(no_punct)
    return no_punct

def words(text):
    return re.findall(r'\w+', text.lower())

WORDS = Counter(words(open('big.txt').read()))

def P(word, N=sum(WORDS.values())):
    "Probability of `word`."
    return WORDS[word] / N

def correction(word):
    "Most probable spelling correction for word."
    return max(candidates(word), key=P)

def candidates(word):
    "Generate possible spelling corrections for word."
    return (known([word]) or known(edits1(word)) or known(edits2(word)) or [word])

def known(words):
    "The subset of `words` that appear in the dictionary of WORDS."
    return set(w for w in words if w in WORDS)

def edits1(word):
    "All edits that are one edit away from `word`."
    letters = 'abcdefghijklmnopqrstuvwxyz'
    splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    deletes = [L + R[1:] for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]
    replaces = [L + c + R[1:] for L, R in splits if R for c in letters]
    inserts = [L + c + R for L, R in splits for c in letters]
    return set(deletes + transposes + replaces + inserts)

def edits2(word):
    "All edits that are two edits away from `word`."
    return (e2 for e1 in edits1(word) for e2 in edits1(e1))

while 1:
    s = input('Input a string: ')
    s = remove_punctuation(s)
    s = s.casefold().split()
    word_check(s)
    print("\n")
    word_check2(s)
    print("\n")

    a = "One possibility:"
    b = " "
    for word in s:
        a = a + correction(word) + b
        if (word != correction(word)):
            print("Other candidates for " + word + ":" + str(known([word])
            or known(edits1(word)) or known(edits2(word)) or [word]))
    print(a)
    print("press q to quit or any other key to enter another string")

    if input() == 'q':
        break





