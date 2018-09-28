import nltk
import sys
import random
from nltk.corpus import PlaintextCorpusReader
from nltk.text import Text
from nltk import word_tokenize
from nltk.util import ngrams
from nltk.probability import FreqDist,ConditionalFreqDist,ConditionalProbDist, LidstoneProbDist
from collections import defaultdict

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

def generateModel(MergedText, MergedWord, ngramModel):
    '''
    This function generates nGram Model.
    '''
    vocabulary = set(MergedText)
    nngrams = list(ngrams(MergedText,ngramModel))
	
    cfd = ConditionalFreqDist()
    ngramSet = set()
    vocabularyOfWords = set()
    fdist = FreqDist()
	
    ProbDictionary = {}
	#Generate conditional frequency distribution 
    
    for ngram in nngrams:
        ngramSet.add(ngram)
        initial_text = tuple(ngram[:-1])
        last_word = ngram[-1]
        cfd[initial_text][last_word] += 1
		
        #Smoothing and generating probabilities using Laplace Algorithm
        laplace_prob = [1.0 * (1+cfd[initial_text][last_word]) / (len(vocabulary)+cfd[initial_text].N())]
        ProbDictionary[initial_text]=laplace_prob
        vocabularyOfWords.add(last_word)
   
    generateSentences(nngrams,cfd,ngramModel)
	    
	
def generateSentences(nngrams,cfd,ngramModel):
    
    for sentence in range(10):
	
        #select a random ngram out of all ngrams generated
        random_ngram = random.choice(nngrams)

        #get rid of the last word from the ngram, "seed" is a list of string with the first (n-1) words
        seed = tuple(random_ngram[:-1])
        #print("seed: ",seed)

        #Predict the next word based on the seed
        predictedWord = cfd[seed].max()
        
        #New text generated by joining seed and the predicted word
        newText = ' '.join(seed) + " " + predictedWord

        #Predict next 15 words for new sentence creation
        for words in range(15):
            #Add the word predicted to the actual seed
            seed+= tuple([predictedWord])

            #Convert the seed to a list
            seedList = list(seed)
            
            #Remove the first element from the seedList to create a new seed...slides to the next seed
            seedList.pop(0)
            
            #Convert the updated seed back to the seed tuple
            seed = tuple(seedList)
			
            #Predict the next word based on the new seed
            predictedWord = cfd[seed].max()
		
            #New Text is created by joining previous newText and the next predicted word
            newText = newText + " " + predictedWord
        print("New Sentence: ", newText)

def main():
    '''
    This is the main function. 
	'''
    
    ngramModel= int(sys.argv[1])
    numSentences = sys.argv[2] 
    numFiles = len(sys.argv)	
    MergedText = read_files(ngramModel,numFiles)

    generateModel(MergedText,ngramModel)
  
    
if __name__ == '__main__':
    main()
