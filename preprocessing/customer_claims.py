import pandas as pd
import STRING
from resources.nif_corrector import id_conversor
from resources import statistics
import datetime
import numpy as np
from models.cluster_model import cluster_analysis


pd.options.display.max_columns = 500


# SOURCE FILE
customer_df = pd.read_csv(STRING.path_db + STRING.file_claim, sep=',', encoding='utf-8', quotechar='"',
                          parse_dates=['poliza_fecha_inicio'])
print(len(customer_df.index))
customer_df = customer_df[customer_df['poliza_fecha_inicio'] >= '2015-01-01']
print(len(customer_df.index))

# BAD ID
'''
customer_df = customer_df[customer_df['cliente_tipo_doc'].isin(['N', 'R', 'P'])]

customer_df['bad_id'] = pd.Series(0, index=customer_df.index)
print(len(customer_df.index))
customer_df['bad_id'] = customer_df.apply(lambda y: id_conversor(y['cliente_tipo_doc'], y['cliente_nif']), axis=1)
customer_df = customer_df[customer_df['bad_id'] != 1]
print(len(customer_df.index))
del customer_df['bad_id']
'''

# SEX VALIDATE
# customer_df['cliente_sexo'] = customer_df['cliente_sexo'].replace('?', -1)
customer_df = customer_df[customer_df['cliente_sexo'] != '?']
customer_df['cliente_sexo'] = customer_df['cliente_sexo'].map(int)
customer_df = customer_df[customer_df['cliente_sexo'].isin([0, 1])]

# FILTER NATIONALITY
# customer_df = customer_df[customer_df['cliente_pais_residencia'] == 'ESPAÑA']

# FILTER CP
customer_df = customer_df[~customer_df['cliente_cp'].astype(str).str.startswith('AD')]  # CP from Andorra
customer_df = customer_df[customer_df['cliente_cp'] != 0]
customer_df = customer_df[customer_df['cliente_cp'] != '0']
# customer_df['cliente_cp'] = customer_df['cliente_cp'].replace('?', -1)
customer_df = customer_df[customer_df['cliente_cp'] != '?']

# REPLACE CLAIMS COST
customer_df.loc[customer_df['cliente_carga_siniestral'] == '?', 'cliente_carga_siniestral'] = 0
customer_df['cliente_carga_siniestral'] = customer_df['cliente_carga_siniestral'].map(float)

# REPLACE BIRTH
customer_df.loc[customer_df['vehiculo_fenacco_conductor1'] == '?', 'vehiculo_fenacco_conductor1'] = customer_df[
    'cliente_fecha_nacimiento']

customer_df.loc[customer_df['cliente_fecha_nacimiento'] == '?', 'cliente_fecha_nacimiento'] = customer_df[
    'vehiculo_fenacco_conductor1']

customer_df = customer_df[customer_df['vehiculo_fenacco_conductor1'] != '?']


# CALCULATE AGE
def calculate_age(birthdate, sep='/'):
    birthdate = datetime.datetime.strptime(birthdate, '%Y' + sep + '%m' + sep + '%d')
    today = datetime.date.today()
    return today.year - birthdate.year - ((today.month, today.day) < (birthdate.month, birthdate.day))


customer_df['cliente_edad'] = customer_df.apply(lambda y: calculate_age(y['vehiculo_fenacco_conductor1']), axis=1)
customer_df.loc[~customer_df['cliente_edad'].between(18, 99, inclusive=True), 'cliente_edad'] = '?'

customer_test = customer_df[['cliente_edad', 'cliente_numero_siniestros', 'cliente_numero_siniestros_auto',
                             'cliente_carga_siniestral']]

customer_test['cliente_edad'] = np.where(customer_test['cliente_edad'] == '?', 1, 0)
customer_test = customer_test.applymap(float)

print(customer_test.groupby(['cliente_edad']).agg({'cliente_numero_siniestros': ['max', 'mean'],
                                                   'cliente_numero_siniestros_auto': 'mean'}))

# We use mean test
statistics.mean_diff_test(customer_df[customer_df['cliente_edad'] != '?'],
                          customer_df[customer_df['cliente_edad'] == '?'], 'cliente_numero_siniestros')

customer_df = customer_df[customer_df['cliente_edad'] != '?']
customer_df['cliente_edad'] = customer_df['cliente_edad'].map(int)

# AGE RANGES
customer_df['cliente_edad_18_30'] = np.where(customer_df['cliente_edad'] <= 30, 1, 0)
customer_df['cliente_edad_30_65'] = np.where(customer_df['cliente_edad'].between(31, 65), 1, 0)
customer_df['cliente_edad_65'] = np.where(customer_df['cliente_edad'] > 65, 1, 0)


# SECOND DRIVER
customer_df.loc[
    customer_df['vehiculo_fenacco_conductor2'] == '?', 'vehiculo_fenacco_conductor2'] = datetime.date.today().strftime(
    '%Y/%m/%d')

customer_df['edad_segundo_conductor'] = customer_df.apply(lambda y: calculate_age(y['vehiculo_fenacco_conductor2']),
                                                          axis=1)
customer_df['edad_segundo_conductor_riesgo'] = np.where(customer_df['edad_segundo_conductor'].between(18, 25), 1, 0)
del customer_df['edad_segundo_conductor']

# LICENSE YEARS FIRST DRIVER
customer_df['antiguedad_permiso'] = customer_df.apply(lambda y: calculate_age(y['vehiculo_fepecon_conductor1']), axis=1)
customer_df['antiguedad_permiso_riesgo'] = np.where(customer_df['antiguedad_permiso'] <= 1, 1, 0)
customer_df.loc[customer_df['antiguedad_permiso'].between(0, 5, inclusive=True), 'antiguedad_permiso_range'] = '[0-5]'
customer_df.loc[customer_df['antiguedad_permiso'].between(6, 10, inclusive=True), 'antiguedad_permiso_range'] = '[6-10]'
customer_df.loc[customer_df['antiguedad_permiso'].between(11, 20, inclusive=True), 'antiguedad_permiso_range'] = '[11-20]'
customer_df.loc[customer_df['antiguedad_permiso'].between(21, 30, inclusive=True), 'antiguedad_permiso_range'] = '[21-30]'
customer_df.loc[customer_df['antiguedad_permiso'] >= 31, 'antiguedad_permiso_range'] = '[31-inf]'


# LICENSE YEARS SECOND DRIVER
customer_df.loc[customer_df['vehiculo_fepecon_conductor2'] == '?', 'vehiculo_fepecon_conductor2'] = '1900/01/01'
customer_df['antiguedad_permiso_segundo'] = customer_df.apply(lambda y: calculate_age(y['vehiculo_fepecon_conductor2']),
                                                              axis=1)
customer_df['antiguedad_permiso_segundo_riesgo'] = np.where(customer_df['antiguedad_permiso_segundo'] <= 1, 1, 0)
customer_df = customer_df.drop(['vehiculo_fepecon_conductor2', 'antiguedad_permiso_segundo'], axis=1)

# VEHICULE USE CODE
customer_df['d_uso_particular'] = np.where(customer_df['vehiculo_uso_desc'].str.contains('PARTICULAR'), 1, 0)
customer_df['d_uso_alquiler'] = np.where(customer_df['vehiculo_uso_desc'].str.contains('ALQUILER'), 1, 0)

# VEHICLE TYPE
tipo_dict = {'ciclomotor': 'PARTICULAR', 'furgoneta': 'FURGONETA', 'camion': 'CAMION', 'autocar': 'AUTOCAR',
             'remolque': 'REMOLQUE', 'agricola': 'AGRICO', 'industrial': 'INDUSTRIAL', 'triciclo': 'TRICICLO'}

for k, v in tipo_dict.items():
    customer_df['d_tipo_' + k] = np.where(customer_df['vehiculo_uso_desc'].str.contains(v), 1, 0)
del tipo_dict

# VEHICLE HEAVY
customer_df['vehiculo_heavy'] = np.where(customer_df['vehiculo_clase_agrupacion_descripcion'].str.contains('>'), 1, 0)

# VEHICLE VALUE
# print(customer_df.vehiculo_valor[~customer_df['vehiculo_valor'].map(np.isreal)])
customer_df = customer_df[customer_df['vehiculo_valor'].map(float) >= 300]

# VEHICLE BONUS
customer_df = customer_df[customer_df['vehiculo_bonus_codigo'] != '?']
customer_df['vehiculo_bonus_codigo'] = customer_df['vehiculo_bonus_codigo'].map(int)

# VEHICLE CATEGORY
cat_dict = {'PRIMERA': '1', 'SEGUNDA': '2', 'TERCERA': '3'}
customer_df['vehiculo_categoria'] = customer_df['vehiculo_categoria'].map(str)
for k, v in cat_dict.items():
    customer_df.loc[customer_df['vehiculo_categoria'].str.contains(k), 'vehiculo_categoria'] = v

# PLATE LICENSE
customer_df = customer_df[customer_df['vehiculo_fecha_mat'].between(1900, 2018, inclusive=True)]
customer_df['antiguedad_vehiculo'] = pd.Series(2018 - customer_df['vehiculo_fecha_mat'], index=customer_df.index)
del customer_df['vehiculo_fecha_mat']

# INTERMEDIARY STATISTICS
customer_df['mediador_riesgo'] = pd.Series(customer_df.mediador_numero_siniestros / customer_df.mediador_numero_polizas,
                                           index=customer_df.index)
customer_df['mediador_riesgo_auto'] = pd.Series(
    customer_df.mediador_numero_siniestros_AUTO / customer_df.mediador_numero_polizas_AUTO, index=customer_df.index)
customer_df['mediador_share_auto'] = pd.Series(
    customer_df.mediador_numero_polizas_AUTO / customer_df.mediador_numero_polizas, index=customer_df.index)

customer_df = customer_df.drop(['mediador_numero_siniestros', 'mediador_clase_intermediario', 'mediador_fecha_alta',
                                'mediador_numero_polizas', 'mediador_numero_polizas_vigor',
                                'mediador_numero_siniestros',
                                'mediador_numero_siniestros_fraude', 'mediador_numero_siniestros_pagados',
                                'mediador_numero_polizas_AUTO',
                                'mediador_numero_polizas_vigor_AUTO', 'mediador_numero_siniestros_AUTO',
                                'mediador_numero_siniestros_fraude_AUTO',
                                'mediador_numero_siniestros_pagados_AUTO', 'mediador_fecha_alta'], axis=1)


# ADDRESS
customer_df['address_complete'] = customer_df['cliente_nombre_via'].map(str) + ' ' + customer_df[
    'cliente_numero_hogar'].map(str) + ', ' + customer_df['cliente_cp'].map(str) + ', Spain'


# GROUPED NATIONALITY
country_file = pd.read_csv(STRING.path_db_aux + STRING.file_country, sep=';', encoding='latin1')
customer_df = pd.merge(customer_df, country_file, left_on='cliente_nacionalidad', right_on='COUNTRY', how='left')
dummy_region = pd.get_dummies(customer_df['REGION'], prefix='cliente_region', dummy_na=True)
customer_df = pd.concat([customer_df, dummy_region], axis=1)
customer_df['cliente_extranjero'] = np.where(customer_df['cliente_nacionalidad'] != 'ESPAÑA', 1, 0)

# POLICY DAYS
customer_df['final_date'] = pd.Series(pd.to_datetime('2017-12-31', format='%Y-%m-%d', errors='coerce'),
                                      index=customer_df.index)
customer_df['policy_days'] = pd.Series((customer_df['final_date'] - customer_df['poliza_fecha_inicio']).dt.days,
                                       index=customer_df.index)


# DROP DUPLICATES BY CUSTOMER/OBJECT
print(len(customer_df.index))
x = customer_df[
    ['cliente_codfiliacion', 'cliente_poliza', 'vehiculo_modelo_desc',
     'cliente_numero_siniestros', 'cliente_carga_siniestral', 'cliente_numero_siniestros_auto',
     'cliente_sexo', 'vehiculo_valor', 'cliente_edad_18_30', 'cliente_edad_30_65', 'cliente_edad_65',
     'edad_segundo_conductor_riesgo', 'antiguedad_permiso', 'antiguedad_permiso_riesgo',
     'antiguedad_permiso_segundo_riesgo',
     'd_uso_particular', 'd_uso_alquiler', 'd_tipo_ciclomotor', 'd_tipo_furgoneta', 'd_tipo_camion', 'd_tipo_autocar',
     'd_tipo_remolque', 'd_tipo_agricola',
     'd_tipo_industrial', 'd_tipo_triciclo', 'antiguedad_vehiculo', 'cliente_extranjero', 'cliente_edad',
     'vehiculo_categoria',
     'vehiculo_heavy', 'mediador_riesgo', 'mediador_riesgo_auto', 'mediador_share_auto', 'REGION',
     'cliente_numero_siniestros_auto_culpa', 'antiguedad_permiso_range', 'cliente_numero_polizas_auto'
     ]]

# x = x.sort_values(by=['cliente_poliza'], ascending=[False])
# x = x.drop_duplicates(subset=['cliente_codfiliacion', 'vehiculo_modelo_desc'], keep='first')

# RISK CLUSTERING BY OBJECT
threshold = x['vehiculo_valor'].quantile(0.99)
print(threshold)
x.loc[x['vehiculo_valor'] > threshold, 'vehiculo_valor'] = threshold
x['vehiculo_valor'] = x['vehiculo_valor'].round()
x['vehiculo_valor'] = x['vehiculo_valor'].map(int)
x['vehiculo_valor'] = pd.cut(x['vehiculo_valor'], range(0, x['vehiculo_valor'].max(), 1000), right=True)
x['vehiculo_valor'] = x['vehiculo_valor'].fillna(x['vehiculo_valor'].max())


x['veh_uso'] = pd.Series('OTRO', index=x.index)
x.loc[x['d_uso_particular'] == 1, 'veh_uso'] = 'PARTICULAR'
x.loc[x['d_uso_alquiler'] == 1, 'veh_uso'] = 'ALQUILER'

x['veh_tipo'] = pd.Series('OTRO', index=x.index)
x.loc[x['d_tipo_ciclomotor'] == 1, 'veh_tipo'] = 'CICLOMOTOR'
x.loc[x['d_tipo_furgoneta'] == 1, 'veh_tipo'] = 'FURGONETA'
x.loc[x['d_tipo_camion'] == 1, 'veh_tipo'] = 'CAMION'
x.loc[x['d_tipo_autocar'] == 1, 'veh_tipo'] = 'AUTOCAR'
x.loc[x['d_tipo_remolque'] == 1, 'veh_tipo'] = 'REMOLQUE'
x.loc[x['d_tipo_agricola'] == 1, 'veh_tipo'] = 'AGRICOLA'
x.loc[x['d_tipo_industrial'] == 1, 'veh_tipo'] = 'INDUSTRIAL'
x.loc[x['d_tipo_triciclo'] == 1, 'veh_tipo'] = 'TRICICLO'

x['counter'] = pd.Series(1, index=x.index)


x_object = x.groupby(
    ['vehiculo_valor', 'vehiculo_categoria',
     'veh_uso', 'veh_tipo', 'vehiculo_heavy', 'antiguedad_vehiculo']).agg(
    {
     'cliente_numero_siniestros_auto': ['mean'], 'cliente_numero_polizas_auto': ['mean'], 'counter': 'count'})

x_object = x_object[x_object[('counter', 'count')] > 5.0]
del x_object['counter']
print('VEHICLE CLUSTER')
cluster_analysis.expl_hopkins(x_object, num_iters=1000)
cluster_analysis.cluster_internal_validation(x_object, n_clusters=10)
cluster_analysis.silhouette_coef(x_object.values, range_n_clusters=range(10, 11, 1))

cluster_analysis.kmeans_plus_plus(x_object, k=10, n_init=42, max_iter=500, drop=None, show_plot=False,
                                  file_name='cluster_veh')


# DROP DUPLICATES CUSTOMERS
# customer_df = customer_df.sort_values(by=['cliente_poliza'], ascending=[False])
# customer_df = customer_df.drop_duplicates(subset=['cliente_codfiliacion'], keep='first')
print(len(customer_df.index))

# DEL VARIABLE
columns_to_drop = ['cliente_fecha_nacimiento', 'cliente_nacionalidad', 'cliente_pais_residencia',
                   'cliente_fechaini_zurich', 'cliente_tipo_doc', 'vehiculo_uso_codigo', 'vehiculo_uso_desc',
                   'vehiculo_clase_codigo', 'vehiculo_clase_descripcion', 'vehiculo_clase_agrupacion_descripcion',
                   'vehiculo_potencia', 'vehiculo_marca_codigo', 'vehiculo_marca_desc',
                   'vehiculo_modelo_codigo', 'vehiculo_modelo_desc', 'vehiculo_bonus_desc',
                   'vehiculo_fenacco_conductor1',
                   'cliente_nombre_via', 'cliente_numero_hogar', 'vehiculo_fenacco_conductor2',
                   'vehiculo_tipo_combustible', 'COUNTRY', 'vehiculo_fepecon_conductor1'
                   ]

customer_df = customer_df.drop(columns_to_drop, axis=1)
customer_df = customer_df[customer_df['cliente_edad'] - customer_df['antiguedad_permiso'] >= 17]

# RISK INTERMEDIARY
x_mediador = customer_df[['mediador_cod_intermediario',  'mediador_riesgo_auto']]

x_mediador = x_mediador.sort_values(by=['mediador_riesgo_auto'], ascending=[False])
x_mediador = x_mediador.drop_duplicates(subset=['mediador_cod_intermediario'], keep='first')
x_mediador = x_mediador.set_index('mediador_cod_intermediario')
for i in x_mediador.columns.values.tolist():
    x_mediador[i] = x_mediador[i] * 100 / x_mediador[i].max()
    x_mediador[i] = x_mediador[i].round()


print('MEDIADOR CLUSTER')
cluster_analysis.expl_hopkins(x_mediador, num_iters=1000)
cluster_analysis.cluster_internal_validation(x_mediador, n_clusters=10)
# cluster_analysis.silhouette_coef(x_mediador.values, range_n_clusters=range(10, 11, 1))


cluster_analysis.kmeans_plus_plus(x_mediador, k=10, n_init=42, max_iter=500, drop=None, show_plot=False,
                                  file_name='cluster_intm')

# RISK CLUSTERING BY CP
cp_risk = customer_df[(customer_df['cliente_numero_siniestros'] < customer_df[
    'cliente_numero_siniestros'].quantile(0.99))]
print(len(customer_df.index))
cp_risk = cp_risk[['cliente_cp',
                   'cliente_numero_siniestros_auto', 'cliente_numero_polizas_auto']]

cp_risk = cp_risk[cp_risk['cliente_cp'] != 0]


cp_risk['cliente_cp'] = cp_risk['cliente_cp'].map(int)
cp_risk = cp_risk.sort_values(by=['cliente_cp'], ascending=True)

cp_risk = cp_risk.groupby(['cliente_cp']).agg({
                                                'cliente_numero_siniestros_auto': ['mean'],
                                                'cliente_numero_polizas_auto': ['mean'],
                                                'cliente_cp': 'count'
                                               })

print(cp_risk)
cp_risk = cp_risk[cp_risk[('cliente_cp', 'count')] > 5.0]
del cp_risk[('cliente_cp', 'count')]

cp_risk = cp_risk.reset_index(drop=False)

print('POSTAL CODE CLUSTER')
cluster_analysis.expl_hopkins(cp_risk.drop(['cliente_cp'], axis=1), num_iters=1000)
cluster_analysis.cluster_internal_validation(cp_risk.drop(['cliente_cp'], axis=1), n_clusters=10)
# cluster_analysis.silhouette_coef(cp_risk.drop(['cliente_cp'], axis=1).values, range_n_clusters=range(10, 11, 1))

cp_risk = cluster_analysis.kmeans_plus_plus(cp_risk, k=10, n_init=42, max_iter=500, drop='cliente_cp', show_plot=False,
                                            file_name='clusters_cp')

cp_risk = cp_risk[[('cliente_cp', ''), 'labels']]
cp_risk = cp_risk.rename(columns={('cliente_cp', ''): 'cliente_cp', 'labels': 'cp_risk'})
customer_df['cliente_cp'] = customer_df['cliente_cp'].map(int)
customer_df = pd.merge(customer_df, cp_risk, on='cliente_cp', how='inner')

# RISK CLUSTERING BY CUSTOMER
# First we normalize THE RISK VARIABLES by AGE IN THE COMPANY

# Risk Variables
target_variables = ['cliente_numero_siniestros_auto', 'cliente_numero_polizas_auto']

for i in target_variables:
    customer_df[i] = customer_df[i] / ((customer_df['policy_days'] + 1)/365)
    customer_df[i] = customer_df[i].round()


customer_df = customer_df[(customer_df['cliente_numero_siniestros_auto'] < customer_df[
    'cliente_numero_siniestros_auto'].quantile(0.99))]

# Cluster at customer
x = customer_df[
    ['cliente_codfiliacion', 'cliente_numero_siniestros', 'cliente_carga_siniestral', 'cliente_numero_siniestros_auto',
     'cliente_sexo', 'cp_risk', 'vehiculo_valor', 'cliente_edad_18_30', 'cliente_edad_30_65', 'cliente_edad_65',
     'edad_segundo_conductor_riesgo', 'antiguedad_permiso', 'antiguedad_permiso_riesgo',
     'antiguedad_permiso_segundo_riesgo',
     'd_uso_particular', 'd_uso_alquiler', 'd_tipo_ciclomotor', 'd_tipo_furgoneta', 'd_tipo_camion', 'd_tipo_autocar',
     'd_tipo_remolque', 'd_tipo_agricola',
     'd_tipo_industrial', 'd_tipo_triciclo', 'antiguedad_vehiculo', 'cliente_extranjero', 'cliente_edad',
     'vehiculo_categoria',
     'vehiculo_heavy', 'mediador_riesgo', 'mediador_riesgo_auto', 'mediador_share_auto', 'REGION',
     'cliente_numero_siniestros_auto_culpa', 'antiguedad_permiso_range', 'cliente_numero_polizas_auto'
     ]]

x['cliente_edad'] = x['cliente_edad'].map(int)
x['cliente_edad'] = pd.cut(x['cliente_edad'], range(x['cliente_edad'].min(), x['cliente_edad'].max(), 5), right=True)
x['cliente_edad'] = x['cliente_edad'].fillna(x['cliente_edad'].max())

x['cliente_numero_siniestros_auto_culpa_share'] = x['cliente_numero_siniestros_auto_culpa'] * 100/ x['cliente_numero_siniestros_auto']
x.loc[x['cliente_numero_siniestros_auto'] == 0, 'cliente_numero_siniestros_auto_culpa_share'] = 0
x['cliente_numero_siniestros_auto_culpa_share'] = x['cliente_numero_siniestros_auto_culpa_share'].round()
x['counter'] = pd.Series(1, index=x.index)
x_customer = x.groupby(
    ['cliente_edad', 'antiguedad_permiso_range', 'edad_segundo_conductor_riesgo', 'cliente_sexo',
     ]).agg(
    {
     'cliente_numero_siniestros_auto': ['mean'],
        'cliente_numero_polizas_auto': ['mean'],
        'counter': ['count']
     })


x_customer = x_customer[x_customer[('counter', 'count')]> 5]
del x_customer['counter']


print('CUSTOMER CLUSTER')
cluster_analysis.expl_hopkins(x_customer, num_iters=1000)
cluster_analysis.cluster_internal_validation(x_customer, n_clusters=10)
# cluster_analysis.silhouette_coef(x_customer.values, range_n_clusters=range(10, 11, 1))

cluster_analysis.kmeans_plus_plus(x_customer, k=10, n_init=42, max_iter=500, drop=None, show_plot=False,
                                  file_name='clusters_customer')




