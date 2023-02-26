import json
import pandas as pd

df_original = pd.read_csv('Translated_Dataset.csv')


df = df_original[df_original['INCLUDED']==0]
df.reset_index(drop=True,inplace=True)
df = df.reset_index(level=0)
df = df.drop(columns=['ID'])
df = df.rename(columns={"index": "ID"}, errors="raise")


# Change NaN values in labels to an empty dict in a string
for i, f in df.iterrows():
    try:
        # f['LABELS'] = eval(f['LABELS'])
        labels_= eval(f['LABELS'])
        labels_ = labels_['labels']
        df.at[i, 'LABELS'] = dict(id=str(f['ID']), text=f['TEXT'], labels=labels_, translation=f['TRANSLATED'])
    except:
        df.at[i, 'LABELS'] = dict(id=str(f['ID']), text=f['TEXT'], labels=[], translation=f['TRANSLATED'])

df['LABELS'].to_json('dataset.json', indent=4)