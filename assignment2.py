'''This Python program called ngram.py will learn an N-gram language model from an arbitrary number of plain text files. The program can generate a given number of sentences based on that N-gram model. 

This program can work for any value of N, and output m sentences as the user requires. Your can run the program as follows:

   ngram.py n m input-file/s
   
n refers to the number of grams and m refers to the number of sentences you want to generate.

for example:
   ngram.py 3 10 'austen-emma.txt' 'austen-persuasion.txt'
   
The .txt files used in this project are from <http://www.gutenberg.org>. Thus, you could chose the files name as follows:

   'austen-emma.txt', 'austen-persuasion.txt', 'austen-sense.txt', 'bible-kjv.txt', 'blake-poems.txt', 'bryant-stories.txt',  
   'burgess-  busterbrown.txt', 'carroll-alice.txt', 'chesterton-ball.txt', 'chesterton-brown.txt', 'chesterton-thursday.txt',  
   'edgeworth-parents.txt', 'melville-moby_dick.txt', 'milton-paradise.txt', 'shakespeare-caesar.txt', 
   'shakespeare-hamlet.txt',   'shakespeare-macbeth.txt', 'whitman-leaves.txt'
'''
import nltk
import sys
import random
from nltk.corpus import PlaintextCorpusReader
from nltk.text import Text
from nltk import word_tokenize
from nltk.util import ngrams
from nltk.probability import FreqDist,ConditionalFreqDist,ConditionalProbDist
from collections import defaultdict
from nltk.corpus import gutenberg

def read_files(ngramModel, numFiles):
    '''
    This function fetches text from all the files provided in the arguments and merges the files.
    This function returns tokens as the output
    '''
    MergedText = []
    for file in range(3,numFiles):
        #Read all the files	
        word =gutenberg.words(sys.argv[file])
        MergedText.extend(word)   #add
        
    return MergedText

def boundaries(nngrams):  
    ''' delete the cross boudary nngrams'''
    new_nngrams=[]
    for i in nngrams:
         if not ( ('end_' in i and i.index('end_')!=len(i)-1) or ('start_' in i and i.index('start_')!=0)):
            new_nngrams.append(i)
    return new_nngrams

def delete_short(words,ngramModel):
    '''1. delete the words from short sentence which words amount is less than n
        2. add the start and end tags on the words list so that each sentance will begin with 'start_' and end with 'end_' '''
    boundary=['.','?','!']  #put the start and end tags when find the end punctations
    new_words=[]
    sentence=[]
    for i in words:       
        sentence.append(i.lower())
        if i in boundary: 
            if len(sentence)>=ngramModel: 
                 sentence.append('end_') #give the end tag if the sentance is ended.
                 new_words.extend(sentence)
            sentence=['start_'] #add a start tag at the begining of a sentence
    return new_words

def generateModel(MergedText, ngramModel, numSentences):
    '''
    This function generates nGram Model.
    '''
    vocabulary = set(MergedText)
	
    MergedText=delete_short(MergedText,ngramModel) # delete the words from short sentence which words amount is less than n

    nngrams = boundaries(list(ngrams(MergedText,ngramModel)))  #generate the nngrams withour cross bouodary
    #nngrams = list(ngrams(MergedText,ngramModel))
    cfd = ConditionalFreqDist()
    ngramSet = set()
    vocabularyOfWords = set()
    fdist = FreqDist()
	
    ProbDictionary = defaultdict(list)
	#Generate conditional frequency distribution 
    
    for ngram in nngrams:
        ngramSet.add(ngram)
        initial_text = tuple(ngram[:-1])
        last_word = ngram[-1]
        cfd[initial_text][last_word] += 1
		
        #Smoothing and generating probabilities using Laplace Algorithm
        laplace_prob = [1.0 * (1+cfd[initial_text][last_word]) / (len(vocabulary)+cfd[initial_text].N())]
        ProbDictionary[initial_text].append(last_word)
        vocabularyOfWords.add(last_word)
   
    generateSentences(nngrams,cfd,ngramModel,ProbDictionary,numSentences)
	    
def find_start_grams(nngrams):
 '''this function found all the grams that can be used at the beginging of a sentence, which has a stat tag'''
    start_grams=[]
    for gram in nngrams:	
          if gram[0]=='start_':
             start_grams.append(gram)
    return start_grams

def generateSentences(nngrams,cfd,ngramModel,ProbDictionary,numSentences):
    
    for sentence in range(int(numSentences)):
	
        #select a random ngram out of all ngrams generated
        start_gram=find_start_grams(nngrams) #find the grams that follows the start mark
        random_ngram = random.choice(start_gram)       
        #get rid of the last word from the ngram, "seed" is a list of string with the first (n-1) words
        
        seed = tuple(random_ngram[:-1])
        #print("seed: ",seed)

        #Predict the next word based on the seed
        #predictedWord = cfd[seed].max()
        for key,value in ProbDictionary.items():
            if(key == seed):
                predictedWord = random.choice(value)
                break
		
        #New text generated by joining seed and the predicted word
        newText = ' '.join(seed) + " " + predictedWord
        end_marks=['.','?','!']
        #Predict next words for new sentence creation
        while (predictedWord!='end_'):  # keep going until find an end mark
            #Add the word predicted to the actual seed
            seed+= tuple([predictedWord])

            #Convert the seed to a list
            seedList = list(seed)
            
            #Remove the first element from the seedList to create a new seed...slides to the next seed
            seedList.pop(0)
            
            #Convert the updated seed back to the seed tuple
            seed = tuple(seedList)
			
            #Predict the next word based on the new seed
            #predictedWord = cfd[seed].max()
			
            for key,value in ProbDictionary.items():
                if(key == seed):
                    predictedWord = random.choice(value)
                    break
		
            #New Text is created by joining previous newText and the next predicted word
            if predictedWord!='end_':
               newText = newText + " " + predictedWord
        print("New Sentence: ", newText[7:])

def main():
    '''
    This is the main function. 
	'''
    
    ngramModel= int(sys.argv[1])
    numSentences = sys.argv[2] 
    numFiles = len(sys.argv)	
    MergedText = read_files(ngramModel,numFiles)
    print("This program generates random sentences based on an Ngram model.")
    print("Command line settings: ngram.py",ngramModel,numSentences)
    generateModel(MergedText,ngramModel,numSentences)
  
    
if __name__ == '__main__':
    main()
