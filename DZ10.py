Ex.1
#Провести дисперсионный анализ для определения того, есть ли различия среднего роста среди 
#взрослых футболистов, хоккеистов и штангистов.
#Даны значения роста в трех группах случайно выбранных спортсменов:
#Футболисты: 173, 175, 180, 178, 177, 185, 183, 182.
#Хоккеисты: 177, 179, 180, 188, 177, 172, 171, 184, 180.
#Штангисты: 172, 173, 169, 177, 166, 180, 178, 177, 172, 166, 170.

import numpy as np
from scipy import stats

football_players = np.array([173, 175, 180, 178, 177, 185, 183, 182])
hockey_players = np.array([177, 179, 180, 188, 177, 172, 171, 184, 180])
weightlifters = np.array([172, 173, 169, 177, 166, 180, 178, 177, 172, 166, 170])


fp_mean = football_players.mean()
hp_mean = hockey_players.mean()
w_mean = weightlifters.mean()

print(f'Средний рост Футболистов = {fp_mean:.2f}, Хоккеистnтов = {hp_mean:.2f}, Штангистов = {w_mean:.2f} ')
print()
print('Проверим статистическую значимость всех отличий.')
concat = np.concatenate([football_players, hockey_players, weightlifters])
print(concat)
concat_mean = concat.mean()
print(concat_mean)

S_F = football_players.shape[0] * (fp_mean - concat_mean) ** 2 + hockey_players.shape[0] * (hp_mean - concat_mean) ** 2 +weightlifters.shape[0] * (w_mean - concat_mean) ** 2
S_res = ((football_players - fp_mean)**2).sum() + ((hockey_players - hp_mean) ** 2).sum() + ((weightlifters - w_mean) ** 2).sum()

print(f'SF = {S_F} S res = {S_res}')
print('Проверим выполнение равенства')
print(S_F + S_res)
print(((concat - concat_mean) ** 2).sum())
print('Запишем оценки дисперсий:')

k = 3
n = football_players.shape[0] + hockey_players.shape[0] + weightlifters.shape[0]

k1 = k - 1
k2 = n - k

sigma_F = S_F / k1
sigma_res = S_res / k2

print(f'{sigma_F:.2f}, {sigma_res:.2f}')

T = sigma_F / sigma_res
print(f'Значение статистики T = {T}')
alpha = 0.05
print('Зафиксируем уровень значимости a = {alpha}. Для него найдём критическое значение F crit')
F_crit = stats.f.ppf(1 - alpha, k1, k2)
print(f'Критерий значения F crit = {F_crit}')
print(f'Видим, что T > F crit ({T:.2f} > {F_crit:.2f}), поэтому заключаем, что отличие среднего роста футболистов, хокеистов и штангистов действительно является статистически значимым.')
