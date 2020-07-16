import csv
import pandas as pd
import string
import matplotlib.pyplot as plt

h = open('dataset\\261K_lyrics_from_MetroLyrics.csv')
read = csv.reader(h)
# skipping the head
read.__next__()
lyrics = {"id_" + str(r[0]): r[5] for r in read}
h.close()

# dictionary final_dict = { song_id : [shingle1_id, shingle2_id, ...] }
# dictionary tuple_dictionary = { tuple : tuple_id}

tuple_dictionary = {}

final_dict = {}

table = str.maketrans('', '', string.punctuation) 

#start = time.time()

last_id = 0
for key in lyrics:
    tokens = lyrics[key].split()
    tokens = [w.lower().translate(table) for w in tokens]
    shingles_list = []
    for i in range(len(tokens) - 3):
        # creating the shingles (lenght = 3)
        tup = tuple(tokens[i : i+3])
        if tup in tuple_dictionary:
            shingles_list.append(tuple_dictionary[tup])
        else:
            tuple_dictionary[tup] = last_id
            shingles_list.append(tuple_dictionary[tup])
            last_id += 1
    final_dict[key] = list(set(shingles_list))
 
        
    
#end = time.time()
#print(round((end - start)/60, 2))

# dropping the empty lists inside the final_dict
keys = []

for key in final_dict:
    if len(final_dict[key]) == 0:
        keys.append(key)

for key in keys:
    if key in final_dict:
        del final_dict[key]

# creating the .tsv file with the song_ids and the list of shingles_id

f = open('dataset\\lyrics_id.tsv', 'w')

for term in final_dict:
    f.write('{0}\t{1}\n'.format(term, final_dict[term]))
    
f.close()


# to compute false-negatives
s_1 = [0.88, 0.9, 0.95, 1] 
# to compute false-positives
s_2 = [0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55,0.5]
r = 10
b = 20


# false_negative
fneg_list = []

for sim in s_1:
    false_neg = (1 -  sim**r)**b
    fneg_list.append(false_neg)

#fneg_list


# false_positive
fpos_list = []

for sim in s_2:
    false_pos = 1 - (1 - sim**r)**b
    fpos_list.append(false_pos)

#fpos_list


#plt.figure(figsize = (10, 8))
#s = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

#r = [10, 5, 15, 5]
#b = [20, 20, 30, 10]
#for i, j in zip(r, b):
#    fs_list = []
#    for sim in s:
#        false_pos = 1 - (1 - sim**i)**j
#        fs_list.append(false_pos)
#    plt.plot(s, fs_list, label = "r = " + str(i) + ", b = " + str(j))
    
#plt.title("S-curve")
#plt.xlabel("Similarity")
#plt.ylabel("Prob")
#plt.legend()
#plt.savefig('s_curve.png')


# importing the output.tsv as pandas dataframe
# this file contains the near duplicates pair candidates

df = pd.read_csv(r'C:\Users\alice\Desktop\HM1_DMT\DMT4BaS__HW_1\DMT4BaS\HW_1\part_2\dataset\output.tsv', sep = "\t")
df2 = df 

# near duplicates list
ND = []
# false positive list
FP = []


for doc_1, doc_2 in zip(df["name_set_1"], df["name_set_2"]):
    a = len(set(final_dict[doc_1]).intersection(set(final_dict[doc_2])))
    b = len(set(final_dict[doc_1]).union(set(final_dict[doc_2])))
    if a/b >= 0.88:
        ND.append((round(a/b, 2), doc_1, doc_2))
    else:
        df2 = df2[(df2.name_set_1 != doc_1) & (df2.name_set_2 != doc_2)]
        FP.append((round(a/b, 2), doc_1, doc_2))
        
#print(len(ND), len(FP))

#df2.to_csv(r'C:\Users\alice\Desktop\HM1_DMT\DMT4BaS__HW_1\DMT4BaS\HW_1\part_2\near_dup.tsv', sep = '\t', index = False)

