import numpy as np
import pandas as pd
import glob
# import seaborn as sns
import matplotlib.pyplot as plt
import json

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, explained_variance_score


table = pd.read_csv('./trainingData_tabular_chunk1.csv')
table_frame = table.drop('gamestate_id', axis=1).drop('decision', axis=1)
pd.options.display.max_columns = 60


df_cards = []
for filename in glob.glob('./*_JSON_*'):
    with open(filename) as fin:
        for record in map(json.loads, fin):
            for card in record['player'].get('hand', []):
                card['turn'] = record['turn']
                card['gamestate_id'] = record['gamestate_id']
                card['hero'] = record['player']['hero']['hero_card_id']
                df_cards.append(card)
json_frame = pd.DataFrame.from_records(df_cards)


unique_frame = json_frame.drop_duplicates(subset=['name'])
col_order = ['name', 'id', 'poisonous', 'shield', 'stealth', 'taunt', 'turn', 'type', 'windfury', 'hero']
unique_ordered_frame = unique_frame[col_order].rename(columns={'hero': "hero_name"})


def get_name_by_id(id):
    return unique_ordered_frame[unique_ordered_frame['hero_name'].eq(id)].iat[0, 0]


table_selected = table[['player.hero_card_id', 'opponent.hero_card_id', 'decision']]
table_selected = table_selected.groupby(['player.hero_card_id', 'opponent.hero_card_id'], as_index=False).mean()
table_selected['player.hero_card_id'] = table_selected['player.hero_card_id'].apply(get_name_by_id)
table_selected['opponent.hero_card_id'] = table_selected['opponent.hero_card_id'].apply(get_name_by_id)
table_selected = table_selected.pivot('player.hero_card_id', 'opponent.hero_card_id')

stat = json_frame[['gamestate_id', 'name']].merge(table[['gamestate_id', 'decision']], how='inner', on='gamestate_id').groupby('name').mean()['decision']


if __name__ == '__main__':
    # Пункт 1
    print(table_frame.describe())
    # Пункт 2
    print(table_selected)
    # Пункт 3
    print(unique_ordered_frame)
    # Пункт 4
    print(stat)
