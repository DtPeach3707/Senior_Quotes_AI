from SQ_Gen import SeniorQuoteGenerator, stack_ragged
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflowjs as tfjs

joke = False
actual = True

if not (joke or actual):
    print('Either joke or actual has to be set to true, or else there will be no data!')

char_lis = ['A']
text_data = []
text_fil_act = open('SeniorQuotes.txt', 'r')
text_fil_joke = open('JokeSeniorQuotes.txt', 'r')

if actual:
    for quote in text_fil_act.readlines():
        print(quote)
        text_data.append(quote)
        for char in quote:
            isOn = False
            for charr in char_lis:
                if charr == char:
                    isOn = True
            if not isOn:
                char_lis.append(char)

if joke:
    for quote in text_fil_joke.readlines():
        print(quote)
        text_data.append(quote)
        for char in quote:
            isOn = False
            for charr in char_lis:
                if charr == char:
                    isOn = True
            if not isOn:
                char_lis.append(char)


char_lis.append('|')
print(len(char_lis))

text_base = []
next_char = []

alphanum = {}
for i in range(len(char_lis)):
  alphanum[char_lis[i]] = i
numalpha = {}

siz = 0
for key in alphanum:
    numalpha[alphanum[key]] = key
    siz += 1

max_num_deletes = 4

print(alphanum)
print(char_lis)

for point in text_data:
    oHE_pipe = [0.0] * siz
    oHE_pipe[alphanum['|']] = 1.0
    vals = [oHE_pipe]
    for text in point:
        oHE = [0.0] * siz
        oHE[alphanum[text]] = 1.0
        vals.append(oHE)
    vals.append(oHE_pipe)
    for i in range(1, len(vals)):
        text_base.append(vals[0:i])
        next_char.append(vals[i])

model = SeniorQuoteGenerator(input_dim=siz)
x_train, x_test, y_train, y_test = train_test_split(text_base, next_char, test_size=0.2)
x_train, x_test = stack_ragged(x_train), stack_ragged(x_test)


tot_episodes = 200
gen_amt = 5
validation = True

episodes = 0
for i in range(int(tot_episodes/5)):
    model.network.fit(x_train, np.array(y_train), validation_data=(x_test, np.array(y_test)), batch_size=50, epochs=5, verbose=True)
    episodes += 5
    print('------ Episode %d ---------' % episodes)
    for j in range(gen_amt):
        model.generate_question(alphanum, numalpha, thresholded=False)

tfjs.converters.save_keras_model(model.network, 'model')
