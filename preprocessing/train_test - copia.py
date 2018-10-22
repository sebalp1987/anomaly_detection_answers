import pandas as pd
import STRING
import resources.process_utils as putils
import seaborn as sns
import matplotlib.pyplot as plot

pd.options.display.max_columns = 500
cluster_cp = pd.read_csv(STRING.path_db_aux + '\\clusters_cp.csv', sep=';', encoding='latin1',
                         dtype={"('cliente_cp', '')": int})

cluster_customer = pd.read_csv(STRING.path_db_aux + '\\clusters_customer.csv', sep=';', encoding='latin1',
                               dtype={'antiguedad_permiso': int, 'edad_segundo_conductor_riesgo':
                                      int, 'cliente_sexo': int, 'REGION': str, 'cp_risk': int})

cluster_veh = pd.read_csv(STRING.path_db_aux + '\\cluster_veh.csv', sep=';', encoding='latin1',
                          dtype={
                              'vehiculo_categoria': int,
                              'veh_uso': str, 'veh_tipo': str,
                              'vehiculo_heavy': int,
                              'antiguedad_vehiculo': int})

cluster_intm = pd.read_csv(STRING.path_db_aux + '\\cluster_intm.csv', sep=';', encoding='latin1',
                           dtype={'oferta_cod_intermediario': int})

cluster_cia = pd.read_csv(STRING.path_db_aux + '\\cia_risk.csv', sep=';', encoding='latin1',
                           dtype={'oferta_sim_cia_actual': int})


offer_processed = pd.read_csv(STRING.path_db_aux + '\\oferta_processed.csv', sep=';', encoding='latin1',
                              dtype={'oferta_tomador_cp': int, 'oferta_cod_intermediario': int,
                                     'antiguedad_permiso': int, 'edad_segundo_conductor_riesgo': int,
                                     'oferta_tomador_sexo': int, 'REGION': str,
                                     'oferta_veh_categoria': int,
                                     'veh_uso': str, 'veh_tipo': str, 'vehiculo_heavy': int})
print(len(offer_processed.index))
drop_var = [
            "('cliente_numero_polizas_auto', 'mean')", "('cliente_numero_siniestros_auto', 'mean')"]

# cluster cp
cluster_cp = cluster_cp.rename(columns={"('cliente_cp', '')": 'oferta_tomador_cp', 'labels': 'cp_risk'})
cluster_cp['weight'] = cluster_cp["('cliente_numero_siniestros_auto', 'mean')"] / cluster_cp["('cliente_numero_polizas_auto', 'mean')"]
cust_risk_relabel = cluster_cp.groupby(['cp_risk']).agg(
    {'weight': 'mean'}).reset_index(
    drop=False)

cust_risk_relabel = cust_risk_relabel.sort_values(by=['weight'], ascending=[True]).reset_index(drop=True)
cust_risk_relabel_cp = cust_risk_relabel.reset_index(drop=False).rename(columns={'index': 'cp_risk_relabel'})
cluster_cp = pd.merge(cluster_cp, cust_risk_relabel_cp[['cp_risk', 'cp_risk_relabel']], how='left',
                            on='cp_risk')
del cluster_cp['cp_risk']


# cluster customer
cluster_customer = cluster_customer.rename(columns={'cliente_sexo': 'oferta_tomador_sexo',
                                                    'labels': 'customer_risk'})
cluster_customer['weight'] = cluster_customer["('cliente_numero_siniestros_auto', 'mean')"] / cluster_customer["('cliente_numero_polizas_auto', 'mean')"]
cust_risk_relabel = cluster_customer.groupby(['customer_risk']).agg(
    {'weight': 'mean'}).reset_index(
    drop=False)

cust_risk_relabel = cust_risk_relabel.sort_values(by=['weight'], ascending=[True]).reset_index(drop=True)
cust_risk_relabel = cust_risk_relabel.reset_index(drop=False).rename(columns={'index': 'customer_risk_relabel'})
cluster_customer = pd.merge(cluster_customer, cust_risk_relabel[['customer_risk', 'customer_risk_relabel']], how='left',
                            on='customer_risk')
del cluster_customer['customer_risk']

cluster_customer = cluster_customer.drop(drop_var, axis=1)

# cluster veh
cluster_veh = cluster_veh.rename(
    columns={'vehiculo_valor': 'oferta_veh_valor', 'vehiculo_categoria': 'oferta_veh_categoria',
             'labels': 'veh_risk'})
cluster_veh['weight'] = cluster_veh["('cliente_numero_siniestros_auto', 'mean')"] / cluster_veh["('cliente_numero_polizas_auto', 'mean')"]
cust_risk_relabel = cluster_veh.groupby(['veh_risk']).agg(
    {'weight': 'mean'}).reset_index(
    drop=False)

cust_risk_relabel = cust_risk_relabel.sort_values(by=['weight'], ascending=[True]).reset_index(drop=True)
cust_risk_relabel_cp = cust_risk_relabel.reset_index(drop=False).rename(columns={'index': 'veh_risk_relabel'})
cluster_veh = pd.merge(cluster_veh, cust_risk_relabel_cp[['veh_risk', 'veh_risk_relabel']], how='left',
                            on='veh_risk')
del cluster_veh['veh_risk']


cluster_veh = cluster_veh.drop(drop_var, axis=1)

# cluster intm
cluster_intm = cluster_intm.rename(columns={'mediador_cod_intermediario':
                                            'oferta_cod_intermediario', 'labels': 'intm_risk'})
int_risk_relabel = cluster_intm.groupby(['intm_risk']).agg({'mediador_riesgo_auto': 'mean'}).reset_index(
    drop=False).sort_values(by=['mediador_riesgo_auto'], ascending=[True]).reset_index(drop=True)
int_risk_relabel = int_risk_relabel.reset_index(drop=False).rename(columns={'index': 'intm_risk_relabel'})
cluster_intm = pd.merge(cluster_intm, int_risk_relabel[['intm_risk', 'intm_risk_relabel']], how='left', on='intm_risk')
del cluster_intm['intm_risk']
# 0 low risk - 9 high risk

# cluster_intm = cluster_intm[['mediador_cod_intermediario', 'intm_risk']]

# CIAS
cluster_cia = cluster_cia.rename(columns={'labels': 'cia_risk'})
cluster_cia_relabel = cluster_cia.groupby(['cia_risk']).agg({'oferta_nivel_sinco_perc': 'mean'}).reset_index(
    drop=False).sort_values(by=['oferta_nivel_sinco_perc'], ascending=[False]).reset_index(drop=True)
cluster_cia_relabel = cluster_cia_relabel.reset_index(drop=False).rename(columns={'index': 'cia_risk_relabel'})
cluster_cia = pd.merge(cluster_cia, cluster_cia_relabel[['cia_risk', 'cia_risk_relabel']], how='left', on='cia_risk')
del cluster_cia['cia_risk']


# MATCH OFERTAS
'''
offer_processed['oferta_sim_cia_actual'] = offer_processed['oferta_sim_cia_actual'].replace('?', -1)
offer_processed['oferta_sim_cia_actual'] = offer_processed['oferta_sim_cia_actual'].map(int)
cluster_cia['oferta_sim_cia_actual'] = cluster_cia['oferta_sim_cia_actual'].map(int)

offer_processed = pd.merge(offer_processed, cluster_cia[['oferta_sim_cia_actual', 'cia_risk_relabel']],
                           how='inner', on='oferta_sim_cia_actual')
offer_processed['cia_risk_relabel'] = offer_processed['cia_risk_relabel'].fillna(-1)
del offer_processed['oferta_sim_cia_actual']
print(len(offer_processed.index))
'''
# MATCH RISK CP
offer_processed['oferta_tomador_cp'] = offer_processed['oferta_tomador_cp'].map(int)
offer_processed = pd.merge(offer_processed, cluster_cp[['oferta_tomador_cp', 'cp_risk_relabel']], on='oferta_tomador_cp', how='left')
del offer_processed['oferta_tomador_cp']
print(len(offer_processed.index))
# MATCH INTM RISK
offer_processed['oferta_cod_intermediario'] = offer_processed['oferta_cod_intermediario'].map(int)
offer_processed = pd.merge(offer_processed, cluster_intm[['oferta_cod_intermediario', 'intm_risk_relabel']],
                           how='inner', on='oferta_cod_intermediario')
del offer_processed['oferta_cod_intermediario']
print(len(offer_processed.index))
# MATCH CUSTOMER RISK
offer_processed = pd.merge(offer_processed, cluster_customer[['cliente_edad', 'antiguedad_permiso_range',
                                                                               'edad_segundo_conductor_riesgo',
                                                                               'oferta_tomador_sexo',
                                                                                'customer_risk_relabel']],
                                                                how='inner', on=['cliente_edad', 'antiguedad_permiso_range',
                                                                               'edad_segundo_conductor_riesgo',
                                                                               'oferta_tomador_sexo'
                                                                               ])
print(len(offer_processed.index))

# MATCH OBJECT
offer_processed = pd.merge(offer_processed, cluster_veh, how='inner', on=['oferta_veh_valor', 'oferta_veh_categoria',
                                                                         'veh_uso', 'veh_tipo', 'antiguedad_vehiculo',
                                                                         'vehiculo_heavy'])
offer_processed.to_csv('test.csv', sep=';', encoding='latin1')
print(len(offer_processed.index))
'''
offer_processed = offer_processed.fillna(-1)
for i in offer_processed.columns.values.tolist():
    offer_processed = offer_processed[offer_processed[i] != -1]
'''
print(len(offer_processed.index))

########################################################################################################################
# ANOMALY DEFINITION
'''
3.	Si el BONUS Simulado es un nivel mejor que el BONUS de consulta SINCO  ->  el BONUS de Emisión será el Simulado, 
siempre que el BONUS de SINCO sea 30% o mejor (KT3F3OT).

4.	Si el BONUS Simulado es más de un nivel mejor que el BONUS de consulta SINCO -> el BONUS de Emisión será el Simulado.
'''
# offer_processed = offer_processed[offer_processed['oferta_bonus_simulacion_perc'] != 0]
print(len(offer_processed.index))
# offer_processed = offer_processed[~((offer_processed['oferta_bonus_simulacion_perc'] ==0.6)&(offer_processed['oferta_nivel_sinco_perc'] ==-0.8))]

# offer_processed = offer_processed[offer_processed['oferta_nivel_sinco_perc'] != -0.8]
# ffer_processed = offer_processed[offer_processed['oferta_nivel_sinco_perc'] != 0.6]
# offer_processed = offer_processed[(offer_processed['oferta_bonus_simulacion_perc'] ==0.6)&(offer_processed['oferta_nivel_sinco_perc'] == -0.8)]
print(len(offer_processed.index))

offer_processed['target'] = pd.Series(0, index=offer_processed.index)

offer_processed['oferta_bonus_simulacion_perc'] = offer_processed['oferta_bonus_simulacion_perc'].map(float) * 100
offer_processed['oferta_nivel_sinco_perc'] = offer_processed['oferta_nivel_sinco_perc'].map(float) * 100

# PLOTS
plot.hist(offer_processed['oferta_bonus_simulacion_perc'])
plot.close()
plot.hist(offer_processed['oferta_nivel_sinco_perc'])
plot.close()
import numpy as np
np.random.seed(19680801)
N = len(offer_processed.index)
area = (25 * np.random.rand(N))**2
plot.scatter(offer_processed['oferta_bonus_simulacion_perc'], offer_processed['oferta_nivel_sinco_perc'], s=area, alpha=0.5)
plot.xlabel('Simulation Bonus')
plot.ylabel('SINCO Bonus')
plot.close()
plot.scatter(offer_processed['oferta_bonus_simulacion_perc'], offer_processed['oferta_bonus_simulacion_perc'] - offer_processed['oferta_nivel_sinco_perc'], s=area, alpha=0.5)
plot.xlabel('Simulation Bonus')
plot.ylabel('Bonus Diff = Sim - SINCO')
plot.show()
plot.close()
offer_processed2 = offer_processed[offer_processed['oferta_bonus_simulacion_perc'] == 60]
plot.hist(offer_processed2['oferta_nivel_sinco_perc'])


offer_processed2 = offer_processed[offer_processed['oferta_nivel_sinco_perc'] < 0]
plot.hist(offer_processed2['oferta_bonus_simulacion_perc'])

# STEP 3

# offer_processed.loc[offer_processed['oferta_nivel_sinco_perc'] < 0, 'target'] = 1
'''
offer_processed.loc[(offer_processed['oferta_bonus_simulacion_perc'] - offer_processed['oferta_nivel_sinco_perc'] == 5) &
                    (offer_processed['oferta_nivel_sinco_perc'] < 30), 'target'] = 1
'''

# STEP 4


offer_processed.loc[(offer_processed['oferta_bonus_simulacion'] -offer_processed['oferta_nivel_sinco'] ==1)&(offer_processed['oferta_nivel_sinco_perc']<30),
                    'target'] = 1

offer_processed.loc[(offer_processed['oferta_bonus_simulacion'] - offer_processed['oferta_nivel_sinco'] > 1),
                    'target'] = 1

print(len(offer_processed.index))
# offer_processed.loc[(offer_processed['oferta_bonus_simulacion_perc'] >= 30)&(offer_processed['oferta_bonus_simulacion_perc'] - offer_processed['oferta_nivel_sinco_perc'] >= 100), 'target'] = 1
# offer_processed.loc[(offer_processed['oferta_nivel_sinco_perc'] < 0), 'target'] = 1
# offer_processed.loc[(offer_processed['oferta_nivel_sinco_perc'].between(-79, 0)), 'target'] = 1
# offer_processed.loc[offer_processed['oferta_nivel_sinco_perc'] == -80, 'target'] = 1
# offer_processed = offer_processed[offer_processed['oferta_bonus_simulacion_perc']==60]
# offer_processed.loc[(offer_processed['oferta_bonus_simulacion_perc'] >= 60)&(offer_processed['oferta_bonus_simulacion_perc'] - offer_processed['oferta_nivel_sinco_perc'] >= 100), 'target'] = 1

#offer_processed.loc[(offer_processed['oferta_bonus_simulacion_perc'].between(0, 20))&(offer_processed['oferta_bonus_simulacion_perc'] - offer_processed['oferta_nivel_sinco_perc'] >= 75)&(offer_processed['oferta_nivel_sinco_perc']<0), 'target'] = 1
#offer_processed.loc[(offer_processed['oferta_bonus_simulacion_perc']>=30)&(offer_processed['oferta_bonus_simulacion_perc'] - offer_processed['oferta_nivel_sinco_perc'] >= 100)&(offer_processed['oferta_nivel_sinco_perc']<0), 'target'] = 1


#####################################################################################################################


# DROP VARIABLES
del_var = ['oferta_veh_valor', 'cliente_edad', 'REGION',
           'veh_uso', 'veh_tipo', 'oferta_poliza.1', 'antiguedad_permiso_range'
           ]

offer_processed = offer_processed.drop(del_var, axis=1)

# FINAL CLEANING

offer_processed = offer_processed.dropna()
offer_processed = offer_processed.rename(columns={'cia_anterior_000': 'd_sin_cia'})
offer_processed['risk_cia'] = offer_processed['cia_anterior_039'] + offer_processed['cia_anterior_040']  # Mutua Madrileña

offer_processed = offer_processed[offer_processed.columns.drop(list(offer_processed.filter(regex='cia_anterior_')))]

offer_processed.to_csv(STRING.path_db_aux + '\\test.csv', sep=';', index=False)
offer_processed = offer_processed.drop(['oferta_nivel_sinco_perc', 'oferta_nivel_sinco', 'oferta_poliza',
                                        'oferta_sim_bonus_rc',
                                        'oferta_sim_bonus_danio'
                                        ], axis=1)
ax = sns.heatmap(offer_processed[['oferta_sim_siniestro_5_anio_culpa', 'oferta_sim_anios_asegurado',
                                  'oferta_sim_antiguedad_cia_actual', 'oferta_sim_siniestro_1_anio_culpa',
                                  'oferta_bonus_simulacion']].corr())


'''
Variables
oferta_id	oferta_tomador_sexo	oferta_tom_cond	oferta_propietario_tom	oferta_propietario_cond	oferta_veh_tara	
oferta_veh_categoria	oferta_veh_puertos	antiguedad_permiso	antiguedad_permiso_riesgo	
edad_segundo_conductor_riesgo	cliente_extranjero	car_ranking	d_uso_particular	
d_uso_alquiler	d_tipo_ciclomotor	d_tipo_furgoneta	d_tipo_camion	d_tipo_autocar	d_tipo_remolque	
d_tipo_agricola	d_tipo_industrial	d_tipo_triciclo	vehiculo_heavy	oferta_sim_siniestro_5_anio_culpa	
oferta_sim_anios_asegurado	oferta_sim_antiguedad_cia_actual	oferta_sim_siniestro_1_anio_culpa	
oferta_bonus_simulacion	oferta_bonus_simulacion_perc	oferta_veh_valor_unitary	cliente_edad_18_30	
cliente_edad_30_65	cliente_edad_65	antiguedad_vehiculo	cliente_region_AMERICADELSUR	cliente_region_EUROPACENTRAL	
cliente_region_EUROPADELNORTE	cliente_region_EUROPADELSUR	cliente_region_EUROPAOCCIDENTAL	cliente_region_EUROPAORIENTAL	
cliente_region_OCEANIA	cliente_region_nan	d_sin_cia	cia_risk_relabel	cp_risk_relabel	cp_control	intm_risk_relabel	
customer_risk_relabel	veh_risk_relabel	target	risk_cia



offer_processed = offer_processed[
    ['oferta_id', 'car_ranking', 'oferta_tomador_sexo', 'oferta_tom_cond', 'antiguedad_permiso_riesgo',
    'edad_segundo_conductor_riesgo',  'oferta_veh_valor_unitary',
     'cliente_edad_18_30', 'antiguedad_vehiculo',  'd_sin_cia',
     'cia_risk_relabel', 'cp_risk_relabel', 'intm_risk_relabel',
     'customer_risk_relabel', 'veh_risk_relabel',
     'target']]

offer_processed = offer_processed[
    ['oferta_id', 'car_ranking', 'oferta_tomador_sexo', 'oferta_tom_cond', 'antiguedad_permiso_riesgo',
    'edad_segundo_conductor_riesgo',  'oferta_veh_valor_unitary',
     'cliente_edad_18_30', 'antiguedad_vehiculo',  'd_sin_cia',
     'cia_risk_relabel', 'cp_risk_relabel', 'intm_risk_relabel',
     'customer_risk_relabel', 'veh_risk_relabel',
    'oferta_propietario_tom', 'oferta_propietario_cond', 'oferta_veh_tara', 'oferta_veh_categoria',
                 'oferta_veh_puertos', 'antiguedad_permiso', 'cliente_extranjero', 'd_uso_particular', 'd_uso_alquiler',
                 'd_tipo_ciclomotor', 'd_tipo_furgoneta', 'd_tipo_camion', 'd_tipo_autocar', 'd_tipo_remolque',
                 'd_tipo_agricola', 'd_tipo_industrial', 'd_tipo_triciclo', 'vehiculo_heavy', 'cliente_edad_30_65',
                 'oferta_veh_plazas',  'oferta_bonus_simulacion_perc', 'target']]


'''
offer_processed = offer_processed.drop(['oferta_veh_tara.1', 'weight', 'oferta_bonus_simulacion'], axis=1)
offer_processed['oferta_id'] = np.random.randint(1, len(offer_processed.index), offer_processed.shape[0])
print(offer_processed.columns.values.tolist())
putils.output_normal_anormal_new(offer_processed)
putils.training_test_valid(offer_processed)
