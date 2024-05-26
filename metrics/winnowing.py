import pygments.token
import pygments.lexers
import hashlib


#Python module pygments is used to tokenize the code files. This module supports most of the popular languages
#http://pygments.org/languages/
#Hence this program can be used to clean up codes written in most languages
def tokenize(text):
    lexer = pygments.lexers.get_lexer_for_filename('asdf.py', text)
    tokens = lexer.get_tokens(text)
    tokens = list(tokens)
    result = []
    lenT = len(tokens)
    count1 = 0    #tag to store corresponding position of each element in original code file
    count2 = 0    #tag to store position of each element in cleaned up code text
    # these tags are used to mark the plagiarized content in the original code files.
    for i in range(lenT):
        if tokens[i][0] == pygments.token.Name and not i == lenT - 1 and not tokens[i + 1][1] == '(':
            result.append(('N', count1, count2))  #all variable names as 'N'
            count2 += 1
        elif tokens[i][0] in pygments.token.Literal.String:
            result.append(('S', count1, count2))  #all strings as 'S'
            count2 += 1
        elif tokens[i][0] in pygments.token.Name.Function:
            result.append(('F', count1, count2))   #user defined function names as 'F'
            count2 += 1
        elif tokens[i][0] == pygments.token.Text or tokens[i][0] in pygments.token.Comment:
            pass   #whitespaces and comments ignored
        else:
            result.append((tokens[i][1], count1, count2))  
            #tuples in result-(each element e.g 'def', its position in original code file, position in cleaned up code/text) 
            count2 += len(tokens[i][1])
        count1 += len(tokens[i][1])

    return result

def toText(arr):
    cleanText = ''.join(str(x[0]) for x in arr)
    return cleanText

#sha-1 encoding is used to generate hash values
def hash(text):
    #this function generates hash values
    hashval = hashlib.sha1(text.encode('utf-8'))
    hashval = hashval.hexdigest()[-4 :]
    hashval = int(hashval, 16)  #using last 16 bits of sha-1 digest
    return hashval

#function to form k-grams out of the cleaned up text
def kgrams(text, k = 25):
    tokenList = list(text)
    n = len(tokenList)
    kgrams = []
    for i in range(n - k + 1):
        kgram = ''.join(tokenList[i : i + k])
        hv = hash(kgram)
        kgrams.append((kgram, hv, i, i + k))  #k-gram, its hash value, starting and ending positions are stored
        #these help in marking the plagiarized content in the original code.
    return kgrams

#function that returns the index at which minimum value of a given list (window) is located
def minIndex(arr):
    minI = 0
    minV = arr[0]
    n = len(arr)
    for i in range(n):
        if arr[i] < minV:
            minV = arr[i]
            minI = i
    return minI

#we form windows of hash values and use min-hash to limit the number of fingerprints
def fingerprints(arr, winSize = 4):
    arrLen = len(arr)
    prevMin = 0
    currMin = 0
    windows = []
    fingerprintList = []
    for i in range(arrLen - winSize):
        win = arr[i: i + winSize]  #forming windows
        windows.append(win)
        currMin = i + minIndex(win)
        if not currMin == prevMin:  #min value of window is stored only if it is not the same as min value of prev window
            fingerprintList.append(arr[currMin])  #reduces the number of fingerprints while maintaining guarantee
            prevMin = currMin  #refer to density of winnowing and guarantee threshold (Stanford paper)

    return fingerprintList

#takes k-gram list as input and returns a list of only hash values
def hashList(arr):
    HL = []
    for i in arr:
        HL.append(i[1])

    return HL

def plagiarismCheck(text1, text2):
    token1 = tokenize(text1)  #from cleanUP.py
    str1 = toText(token1)
    token2 = tokenize(text2)
    str2 = toText(token2)
    kGrams1 = kgrams(str1)  #stores k-grams, their hash values and positions in cleaned up text
    kGrams2 = kgrams(str2)
    HL1 = hashList(kGrams1)  #hash list derived from k-grams list
    HL2 = hashList(kGrams2)
    fpList1 = fingerprints(HL1)
    fpList2 = fingerprints(HL2)
    start = []   #to store the start values corresponding to matching fingerprints
    end = []   #to store end values
    code = text1  #original code
    newCode = ""   #code with marked plagiarized content
    points = []
    for i in fpList1:
        for j in fpList2:
            if i == j:   #fingerprints match
                flag = 0
                match = HL1.index(i)   #index of matching fingerprints in hash list, k-grams list
                newStart = kGrams1[match][2]   #start position of matched k-gram in cleaned up code
                newEnd = kGrams1[match][3]   #end position
                for k in token1:
                    if k[2] == newStart:   #linking positions in cleaned up code to original code
                        startx = k[1]
                        flag = 1
                    if k[2] == newEnd:
                        endx = k[1]
                if flag == 1:
                    points.append([startx, endx])
    points.sort(key = lambda x: x[0])
    points = points[1:]
    mergedPoints = []
    if len(points) > 0:
        mergedPoints.append(points[0])
    for i in range(1, len(points)):
        last = mergedPoints[len(mergedPoints) - 1]
        if points[i][0] >= last[0] and points[i][0] <= last[1]: #merging overlapping regions
            if points[i][1] > last[1]:
                mergedPoints = mergedPoints[: len(mergedPoints)-1]
                mergedPoints.append([last[0], points[i][1]])
            else:
                pass
        else:
            mergedPoints.append(points[i])

    plagCount = 0
    for i in range(len(mergedPoints)):
        if mergedPoints[i][1] > mergedPoints[i][0]:
            plagCount += mergedPoints[i][1] - mergedPoints[i][0]
    return plagCount / len(code)



if __name__ == '__main__':
    import jsonlines
    from datasets import load_dataset

    def load_jsonl(filename):
        results = []
        with jsonlines.open(filename) as reader:
            for obj in reader:
                results.append(obj)
        return results


    generations = load_jsonl("generations/github_CodeLlama-7b-Instruct_num3000_min100.jsonl")


    code_data = load_dataset("LLM-PBE/github-python").shuffle()

    overall_score=[]
    for i, item in enumerate(generations):
        idx= item['idx']
        prompt= item['prompt']
        output= item['output']
        ground_truth= code_data['train'][idx]['body']
        print("\n******\n")
        print(output)
        print("\n******\n")
        print(ground_truth)
        print("\n******\n")
        score = plagiarismCheck(ground_truth, ground_truth)
        print(score)
        overall_score.append(score)
        break
