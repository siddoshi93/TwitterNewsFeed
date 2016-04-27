from preprocess import get_text,english,parse,valid,Tokenizer

politics =[]
sports =[]
technology =[]
finance =[]
entertainment = []


dataset = []
target = []


f = open('data/politics', 'rU')
for line in f:
    politics.append(line)      
politics = map(get_text,filter(english,map(parse,politics)))

f = open('data/sports', 'rU')
for line in f:
    sports.append(line)       
sports = map(get_text,filter(english,map(parse,sports)))

f = open('data/technology', 'rU')
for line in f:
    technology.append(line)       
technology = map(get_text,filter(english,map(parse,technology)))

f = open('data/entertainment', 'rU')
for line in f:
    entertainment.append(line)       
entertainment = map(get_text,filter(english,map(parse,entertainment)))

f = open('data/finance', 'rU')
for line in f:
    finance.append(line)       
finance = map(get_text,filter(english,map(parse,finance)))



politics = list(set(politics))
sports = list(set(sports))
technology = list(set(technology))
entertainment = list(set(entertainment))
finance = list(set(finance))


for tweet in politics:
    target.append(0)

for tweet in sports:
    target.append(1)

for tweet in technology:
    target.append(2)

for tweet in entertainment:
    target.append(3)

for tweet in finance:
    target.append(4)


dataset = politics + sports +technology + entertainment + finance
