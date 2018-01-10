import time
import os
from heapq import heappush
from heapq import heappop

class Word():
    def __init__(self,time,eng,ch,interval):
        self.time=time
        self.eng=eng
        self.ch=ch
        self.interval=interval

    def __lt__(self,other):
        return self.time<other.time
    
    def string(self):
        return '{} {} {} {}'.format(self.time,self.eng,self.ch,self.interval)

def Load(words):
    fileIn=open('words.sav','r')
    while 1:
        s=fileIn.readline()
        l=len(s)
        if l==0:
            break
        i=0
        t=''
        while i<l and s[i]!=' ':
            t+=s[i]
            i+=1
        time=int(t)
        t=''
        i+=1
        while i<l and s[i]!=' ':
            t+=s[i]
            i+=1
        eng=t
        t=''
        i+=1
        while i<l and s[i]!=' ':
            t+=s[i]
            i+=1
        ch=t
        t=''
        i+=1
        while i<l and s[i]!=' ':
            t+=s[i]
            i+=1
        interval=int(t)
        heappush(words,Word(time,eng,ch,interval))
    fileIn.close()

def Write(words):
    fileOut=open('words.sav','w')
    for e in words:
        fileOut.write(e.string()+'\n')
    fileOut.close()

if __name__=='__main__':
    words=[]
    Load(words)
    c=''
    while c!='3':
        c=input('Type \'1\' to append new word, type \'2\' to get into Test, type \'3\' to end\n')
        if c=='1':
            while 1:
                eng=input('English:')
                if eng == '>>':
                    break
                ch=input('Chinese:')
                heappush(words,Word(int(time.time()+1),eng,ch,4))
        if c=='2':
            while len(words)>0 and words[0].time<int(time.time()):
                print('Chinese:',words[0].ch)
                eng=input('English: ')
                if eng == '>>':
                    break
                if eng==words[0].eng:
                    print('Right!')
                    word=words[0]
                    heappop(words)
                    word.interval*=4
                    word.time=int(time.time())+word.interval
                    heappush(words,word)
                else:
                    print('Wrong!The right answer is',words[0].eng)
                    word=words[0]
                    heappop(words)
                    if word.interval > 4:
                        word.interval//=4
                    word.time=int(time.time())+word.interval
                    heappush(words,word)
                    input()
                time.sleep(1)
                os.system('cls')
            print('There is no word which needs to be tested now.And the closeast time able to test is',time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(words[0].time)))
    Write(words)